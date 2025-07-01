package com.perphproctor.recommendation;

import org.apache.hadoop.yarn.api.records.NodeId;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * ConfidenceChecker implements Algorithm 2 from the paper:
 * "Node schedulability checking based on confidence".
 * 
 * It analyzes recommendation scores to determine appropriate thresholds
 * for node acceptance decisions, ensuring sufficient nodes remain available
 * for batch task placement while protecting LRA performance.
 */
public class ConfidenceChecker {
    
    private static final Logger LOG = LoggerFactory.getLogger(ConfidenceChecker.class);
    
    // Reference to recommendation calculator
    private final RecommendationCalculator recommendationCalculator;
    
    // Cache of node acceptance decisions
    private final Map<NodeId, Boolean> nodeAcceptanceCache;
    
    // Last cache clear time
    private long lastCacheClearTime;
    
    // Cache validity period (ms)
    private static final long CACHE_VALIDITY_PERIOD = 10000; // 10 seconds
    
    /**
     * Constructor
     * 
     * @param recommendationCalculator Reference to recommendation calculator
     */
    public ConfidenceChecker(RecommendationCalculator recommendationCalculator) {
        this.recommendationCalculator = recommendationCalculator;
        this.nodeAcceptanceCache = new ConcurrentHashMap<>();
        this.lastCacheClearTime = System.currentTimeMillis();
        
        LOG.info("ConfidenceChecker initialized");
    }
    
    /**
     * Check if a node should accept a new task based on recommendation scores
     * 
     * @param nodeId Node identifier
     * @return true if node should accept task, false otherwise
     */
    public boolean shouldNodeAcceptTask(NodeId nodeId) {
        if (nodeId == null) {
            return false;
        }
        
        // Check cache first
        if (isCacheValid()) {
            Boolean cachedResult = nodeAcceptanceCache.get(nodeId);
            if (cachedResult != null) {
                return cachedResult;
            }
        } else {
            // Clear expired cache
            clearCache();
        }
        
        // Get all recommendation scores
        Map<NodeId, Double> scores = recommendationCalculator.getNodeRecommendationScores();
        
        // Get this node's score
        double nodeScore = scores.getOrDefault(nodeId, 0.5);
        
        // Get acceptance threshold
        double threshold = calculateAcceptanceThreshold(scores);
        
        // Determine acceptance
        boolean shouldAccept = nodeScore >= threshold;
        
        // Cache the result
        nodeAcceptanceCache.put(nodeId, shouldAccept);
        
        LOG.debug("Node {} acceptance: {} (score={}, threshold={})", 
                 nodeId, shouldAccept, nodeScore, threshold);
        
        return shouldAccept;
    }
    
    /**
     * Calculate appropriate acceptance threshold based on score distribution
     * 
     * @param scores Map of node IDs to recommendation scores
     * @return Calculated threshold
     */
    public double calculateAcceptanceThreshold(Map<NodeId, Double> scores) {
        // If no scores available, use default threshold
        if (scores == null || scores.isEmpty()) {
            return 0.5;
        }
        
        // Calculate mean and standard deviation
        List<Double> scoreValues = new ArrayList<>(scores.values());
        double mean = calculateMean(scoreValues);
        double stdDev = calculateStdDev(scoreValues, mean);
        
        // Count nodes above thresholds
        int threshold1Count = 0; // Count above mean
        int threshold2Count = 0; // Count above (mean - stdDev)
        int totalCount = scoreValues.size();
        
        for (Double score : scoreValues) {
            if (score > mean) {
                threshold1Count++;
            }
            if (score > (mean - stdDev)) {
                threshold2Count++;
            }
        }
        
        // Determine appropriate threshold based on algorithm 2
        double threshold;
        
        if (threshold1Count > totalCount / 2) {
            // If majority of nodes have score above mean, use mean as threshold
            threshold = mean;
            LOG.debug("Using mean as threshold: {}", threshold);
        } else if (threshold2Count > totalCount / 2) {
            // If majority of nodes have score above (mean - stdDev), use that as threshold
            threshold = mean - stdDev;
            LOG.debug("Using (mean - stdDev) as threshold: {}", threshold);
        } else {
            // Otherwise, use (mean - 2*stdDev) as threshold
            threshold = mean - (2 * stdDev);
            // Ensure threshold isn't too low
            threshold = Math.max(0.1, threshold);
            LOG.debug("Using (mean - 2*stdDev) as threshold: {}", threshold);
        }
        
        return threshold;
    }
    
    /**
     * Calculate mean of a list of values
     * 
     * @param values List of double values
     * @return Mean value
     */
    private double calculateMean(List<Double> values) {
        if (values == null || values.isEmpty()) {
            return 0.0;
        }
        
        double sum = 0.0;
        for (Double value : values) {
            sum += value;
        }
        
        return sum / values.size();
    }
    
    /**
     * Calculate standard deviation of a list of values
     * 
     * @param values List of double values
     * @param mean Pre-calculated mean value
     * @return Standard deviation
     */
    private double calculateStdDev(List<Double> values, double mean) {
        if (values == null || values.isEmpty()) {
            return 0.0;
        }
        
        double sumSquaredDifferences = 0.0;
        for (Double value : values) {
            double diff = value - mean;
            sumSquaredDifferences += diff * diff;
        }
        
        return Math.sqrt(sumSquaredDifferences / values.size());
    }
    
    /**
     * Check if the cache is still valid
     * 
     * @return true if cache is valid, false if expired
     */
    private boolean isCacheValid() {
        long currentTime = System.currentTimeMillis();
        return (currentTime - lastCacheClearTime) < CACHE_VALIDITY_PERIOD;
    }
    
    /**
     * Clear the acceptance cache
     */
    private void clearCache() {
        nodeAcceptanceCache.clear();
        lastCacheClearTime = System.currentTimeMillis();
        LOG.debug("Cleared node acceptance cache");
    }
    
    /**
     * Force clear of the cache
     */
    public void forceClearCache() {
        clearCache();
        LOG.info("Forced clearing of node acceptance cache");
    }
    
    /**
     * Get the current acceptance cache validity state
     * 
     * @return true if cache is valid, false if expired
     */
    public boolean isCacheCurrentlyValid() {
        return isCacheValid();
    }
    
    /**
     * Get the number of cached acceptance decisions
     * 
     * @return Cache size
     */
    public int getCacheSize() {
        return nodeAcceptanceCache.size();
    }
    
    /**
     * Get cache statistics
     * 
     * @return Map containing cache statistics
     */
    public Map<String, Object> getCacheStatistics() {
        Map<String, Object> stats = new java.util.HashMap<>();
        
        stats.put("cacheSize", nodeAcceptanceCache.size());
        stats.put("cacheAge", System.currentTimeMillis() - lastCacheClearTime);
        stats.put("cacheValid", isCacheValid());
        
        // Count true/false decisions
        int acceptCount = 0;
        int rejectCount = 0;
        
        for (Boolean accept : nodeAcceptanceCache.values()) {
            if (accept) {
                acceptCount++;
            } else {
                rejectCount++;
            }
        }
        
        stats.put("nodeAcceptCount", acceptCount);
        stats.put("nodeRejectCount", rejectCount);
        
        return stats;
    }
    
    /**
     * Calculate different thresholds for multi-level decision making
     * 
     * @return Array of thresholds [strict, normal, relaxed]
     */
    public double[] calculateMultiLevelThresholds() {
        Map<NodeId, Double> scores = recommendationCalculator.getNodeRecommendationScores();
        
        double baseThreshold = calculateAcceptanceThreshold(scores);
        
        // Calculate multi-level thresholds
        double strictThreshold = baseThreshold + 0.1;
        double normalThreshold = baseThreshold;
        double relaxedThreshold = baseThreshold - 0.1;
        
        // Ensure thresholds are within valid range
        strictThreshold = Math.min(0.9, strictThreshold);
        relaxedThreshold = Math.max(0.1, relaxedThreshold);
        
        return new double[] { strictThreshold, normalThreshold, relaxedThreshold };
    }
}