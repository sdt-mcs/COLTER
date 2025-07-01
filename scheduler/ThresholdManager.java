package com.perphproctor.scheduler;

import org.apache.hadoop.yarn.api.records.NodeId;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

/**
 * ThresholdManager implements the confidence-based node schedulability checking
 * algorithm (Algorithm 2 in the paper). It dynamically adjusts acceptance thresholds
 * using statistical analysis of node recommendation scores.
 * 
 * This component ensures sufficient nodes remain available for batch task placement
 * while protecting LRA performance, using multi-level thresholds based on normal
 * distribution properties.
 */
public class ThresholdManager {
    
    private static final Logger LOG = LoggerFactory.getLogger(ThresholdManager.class);
    
    // Reference to the main scheduler
    private final AdaptiveScheduler adaptiveScheduler;
    
    // Thread-safe reference to current threshold
    private final AtomicReference<Double> currentThreshold;
    
    // Last update time
    private long lastUpdateTime;
    
    // Update interval
    private static final long UPDATE_INTERVAL_MS = 10000; // 10 seconds
    
    /**
     * Constructor
     * 
     * @param adaptiveScheduler Reference to the adaptive scheduler
     */
    public ThresholdManager(AdaptiveScheduler adaptiveScheduler) {
        this.adaptiveScheduler = adaptiveScheduler;
        this.currentThreshold = new AtomicReference<>(0.5); // Default threshold
        this.lastUpdateTime = System.currentTimeMillis();
        
        LOG.info("ThresholdManager initialized with default threshold of 0.5");
    }
    
    /**
     * Calculate the appropriate acceptance threshold based on current node scores
     * 
     * @return Current acceptance threshold value
     */
    public double calculateAcceptanceThreshold() {
        long currentTime = System.currentTimeMillis();
        
        // Only recalculate threshold periodically to avoid overhead
        if (currentTime - lastUpdateTime < UPDATE_INTERVAL_MS) {
            return currentThreshold.get();
        }
        
        // Get current recommendation scores
        Map<NodeId, Double> scores = adaptiveScheduler.getNodeRecommendationScores();
        
        // If no scores available, use default
        if (scores.isEmpty()) {
            return 0.5;
        }
        
        // Convert to list for statistical calculations
        List<Double> scoresList = new ArrayList<>(scores.values());
        Collections.sort(scoresList);
        
        // Calculate mean and standard deviation
        double mean = calculateMean(scoresList);
        double stdDev = calculateStandardDeviation(scoresList, mean);
        
        // Counters for thresholds
        int threshold1Count = 0; // Count of scores above mean
        int threshold2Count = 0; // Count of scores above (mean - stdDev)
        int nodeCount = scoresList.size();
        
        // Count nodes above thresholds
        for (Double score : scoresList) {
            if (score > mean) {
                threshold1Count++;
            }
            if (score > (mean - stdDev)) {
                threshold2Count++;
            }
        }
        
        // Determine appropriate threshold
        double newThreshold;
        
        if (threshold1Count > nodeCount / 2) {
            // Majority of nodes have scores above mean
            newThreshold = mean;
            LOG.debug("Setting threshold to mean: {}", newThreshold);
        } else if (threshold2Count > nodeCount / 2) {
            // Majority of nodes have scores above (mean - stdDev)
            newThreshold = mean - stdDev;
            LOG.debug("Setting threshold to mean - stdDev: {}", newThreshold);
        } else {
            // Few nodes have high scores, use lower threshold
            newThreshold = mean - 2 * stdDev;
            // Ensure threshold doesn't go too low
            newThreshold = Math.max(0.1, newThreshold);
            LOG.debug("Setting threshold to mean - 2*stdDev: {}", newThreshold);
        }
        
        // Update current threshold
        currentThreshold.set(newThreshold);
        lastUpdateTime = currentTime;
        
        LOG.info("Updated acceptance threshold to {} (mean={}, stdDev={})", 
                 newThreshold, mean, stdDev);
        
        return newThreshold;
    }
    
    /**
     * Check if a node is acceptable based on current threshold
     * 
     * @param nodeId Node identifier
     * @return true if node should accept tasks
     */
    public boolean isNodeAcceptable(NodeId nodeId) {
        if (nodeId == null) {
            return false;
        }
        
        // Get node's recommendation score
        Double score = adaptiveScheduler.getNodeRecommendationScores().getOrDefault(nodeId, 0.5);
        
        // Get current threshold
        double threshold = currentThreshold.get();
        
        return score >= threshold;
    }
    
    /**
     * Calculate the mean of a list of values
     * 
     * @param values List of values
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
     * Calculate the standard deviation of a list of values
     * 
     * @param values List of values
     * @param mean Mean value (pre-calculated)
     * @return Standard deviation
     */
    private double calculateStandardDeviation(List<Double> values, double mean) {
        if (values == null || values.isEmpty()) {
            return 0.0;
        }
        
        double sumSquaredDiff = 0.0;
        for (Double value : values) {
            double diff = value - mean;
            sumSquaredDiff += diff * diff;
        }
        
        return Math.sqrt(sumSquaredDiff / values.size());
    }
    
    /**
     * Get the current acceptance threshold
     * 
     * @return Current threshold value
     */
    public double getCurrentThreshold() {
        return currentThreshold.get();
    }
    
    /**
     * Manually set the acceptance threshold
     * 
     * @param threshold New threshold value (0-1)
     */
    public void setThreshold(double threshold) {
        if (threshold < 0.0 || threshold > 1.0) {
            LOG.warn("Invalid threshold value: {}. Must be between 0 and 1.", threshold);
            return;
        }
        
        currentThreshold.set(threshold);
        lastUpdateTime = System.currentTimeMillis();
        LOG.info("Manually updated acceptance threshold to {}", threshold);
    }
    
    /**
     * Calculate statistical information about node scores
     * 
     * @return Map containing statistical information
     */
    public Map<String, Double> getStatistics() {
        Map<NodeId, Double> scores = adaptiveScheduler.getNodeRecommendationScores();
        List<Double> scoresList = new ArrayList<>(scores.values());
        
        double mean = calculateMean(scoresList);
        double stdDev = calculateStandardDeviation(scoresList, mean);
        
        // Sort for percentile calculations
        Collections.sort(scoresList);
        
        // Calculate percentiles
        double p25 = 0.0;
        double p50 = 0.0;
        double p75 = 0.0;
        
        if (!scoresList.isEmpty()) {
            int size = scoresList.size();
            p25 = scoresList.get(Math.max(0, (int)(size * 0.25) - 1));
            p50 = scoresList.get(Math.max(0, (int)(size * 0.50) - 1));
            p75 = scoresList.get(Math.max(0, (int)(size * 0.75) - 1));
        }
        
        // Create statistics map
        Map<String, Double> stats = new java.util.HashMap<>();
        stats.put("mean", mean);
        stats.put("standardDeviation", stdDev);
        stats.put("p25", p25);
        stats.put("p50", p50);
        stats.put("p75", p75);
        stats.put("currentThreshold", currentThreshold.get());
        
        return stats;
    }
    
    /**
     * Dynamically adjust the threshold based on cluster conditions
     * 
     * @param clusterLoad Current cluster load (0-1)
     * @param qosViolationRate Current QoS violation rate (0-1)
     */
    public void adjustThresholdDynamically(double clusterLoad, double qosViolationRate) {
        double currentThreshold = this.currentThreshold.get();
        double newThreshold = currentThreshold;
        
        // High cluster load requires stricter threshold
        if (clusterLoad > 0.8) {
            // Increase threshold to be more selective
            newThreshold += 0.1;
        } else if (clusterLoad < 0.3) {
            // Decrease threshold to utilize resources
            newThreshold -= 0.1;
        }
        
        // QoS violations require stricter threshold
        if (qosViolationRate > 0.05) {
            // Increase threshold to protect LRA performance
            newThreshold += 0.15;
        }
        
        // Clamp threshold to valid range
        newThreshold = Math.max(0.1, Math.min(0.9, newThreshold));
        
        // Only update if significant change
        if (Math.abs(newThreshold - currentThreshold) > 0.05) {
            setThreshold(newThreshold);
            LOG.info("Dynamically adjusted threshold to {} based on cluster load {} and QoS violation rate {}", 
                    newThreshold, clusterLoad, qosViolationRate);
        }
    }
    
    /**
     * Calculate multi-level thresholds for different scheduling priorities
     * 
     * @return Array of thresholds [high, medium, low]
     */
    public double[] calculateMultiLevelThresholds() {
        double baseThreshold = currentThreshold.get();
        
        // Calculate thresholds for different priority levels
        double highPriorityThreshold = baseThreshold + 0.1;  // Stricter for high priority
        double mediumPriorityThreshold = baseThreshold;      // Base for medium priority
        double lowPriorityThreshold = baseThreshold - 0.1;   // More relaxed for low priority
        
        // Clamp thresholds to valid range
        highPriorityThreshold = Math.min(0.9, highPriorityThreshold);
        lowPriorityThreshold = Math.max(0.1, lowPriorityThreshold);
        
        return new double[] { highPriorityThreshold, mediumPriorityThreshold, lowPriorityThreshold };
    }
    
    /**
     * Reset the threshold manager to default state
     */
    public void reset() {
        currentThreshold.set(0.5);
        lastUpdateTime = System.currentTimeMillis();
        LOG.info("ThresholdManager reset to default state");
    }
}