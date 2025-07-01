package com.perphproctor.recommendation;

import org.apache.hadoop.yarn.api.records.NodeId;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * ScoreNormalizer processes raw recommendation scores to ensure
 * consistent and comparable values across nodes. It implements
 * various normalization strategies to handle different cluster conditions
 * and score distributions.
 */
public class ScoreNormalizer {
    
    private static final Logger LOG = LoggerFactory.getLogger(ScoreNormalizer.class);
    
    // Normalization strategies
    public enum NormalizationStrategy {
        MIN_MAX,         // Scale to [0,1] using min-max normalization
        Z_SCORE,         // Standardize using z-scores
        SIGMOID,         // Apply sigmoid transformation
        PERCENTILE,      // Normalize based on percentile rank
        NONE             // No normalization
    }
    
    // Current normalization strategy
    private NormalizationStrategy currentStrategy;
    
    /**
     * Constructor with default normalization strategy
     */
    public ScoreNormalizer() {
        this(NormalizationStrategy.MIN_MAX);
    }
    
    /**
     * Constructor with specified normalization strategy
     * 
     * @param strategy Normalization strategy to use
     */
    public ScoreNormalizer(NormalizationStrategy strategy) {
        this.currentStrategy = strategy;
        LOG.info("ScoreNormalizer initialized with strategy: {}", strategy);
    }
    
    /**
     * Normalize recommendation scores
     * 
     * @param scores Map of node IDs to raw recommendation scores
     * @return Map of node IDs to normalized scores
     */
    public Map<NodeId, Double> normalizeScores(Map<NodeId, Double> scores) {
        if (scores == null || scores.isEmpty()) {
            return new HashMap<>();
        }
        
        // Extract score values
        List<Double> scoreValues = new ArrayList<>(scores.values());
        
        // Apply normalization based on current strategy
        Map<NodeId, Double> normalizedScores = new HashMap<>();
        
        switch (currentStrategy) {
            case MIN_MAX:
                normalizedScores = applyMinMaxNormalization(scores, scoreValues);
                break;
            case Z_SCORE:
                normalizedScores = applyZScoreNormalization(scores, scoreValues);
                break;
            case SIGMOID:
                normalizedScores = applySigmoidNormalization(scores);
                break;
            case PERCENTILE:
                normalizedScores = applyPercentileNormalization(scores, scoreValues);
                break;
            case NONE:
                normalizedScores = new HashMap<>(scores);
                break;
        }
        
        LOG.debug("Normalized scores using strategy {}: {} raw scores -> {} normalized scores",
                 currentStrategy, scores.size(), normalizedScores.size());
        
        return normalizedScores;
    }
    
    /**
     * Apply min-max normalization to scale scores to [0,1] range
     * 
     * @param scores Map of node IDs to raw scores
     * @param scoreValues List of score values
     * @return Map of node IDs to normalized scores
     */
    private Map<NodeId, Double> applyMinMaxNormalization(Map<NodeId, Double> scores, 
                                                       List<Double> scoreValues) {
        Map<NodeId, Double> normalized = new HashMap<>();
        
        // Find min and max values
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        
        for (Double score : scoreValues) {
            min = Math.min(min, score);
            max = Math.max(max, score);
        }
        
        // Handle case where all scores are the same
        if (max == min) {
            for (Map.Entry<NodeId, Double> entry : scores.entrySet()) {
                normalized.put(entry.getKey(), 0.5); // Default to middle value
            }
            return normalized;
        }
        
        // Apply min-max normalization
        for (Map.Entry<NodeId, Double> entry : scores.entrySet()) {
            double normalizedScore = (entry.getValue() - min) / (max - min);
            normalized.put(entry.getKey(), normalizedScore);
        }
        
        return normalized;
    }
    
    /**
     * Apply Z-score normalization (standardization)
     * 
     * @param scores Map of node IDs to raw scores
     * @param scoreValues List of score values
     * @return Map of node IDs to normalized scores
     */
    private Map<NodeId, Double> applyZScoreNormalization(Map<NodeId, Double> scores, 
                                                       List<Double> scoreValues) {
        Map<NodeId, Double> normalized = new HashMap<>();
        
        // Calculate mean
        double sum = 0.0;
        for (Double score : scoreValues) {
            sum += score;
        }
        double mean = sum / scoreValues.size();
        
        // Calculate standard deviation
        double sumSquaredDiff = 0.0;
        for (Double score : scoreValues) {
            double diff = score - mean;
            sumSquaredDiff += diff * diff;
        }
        double stdDev = Math.sqrt(sumSquaredDiff / scoreValues.size());
        
        // Handle case where stdDev is zero
        if (stdDev == 0) {
            for (Map.Entry<NodeId, Double> entry : scores.entrySet()) {
                normalized.put(entry.getKey(), 0.5); // Default to middle value
            }
            return normalized;
        }
        
        // Apply Z-score normalization
        for (Map.Entry<NodeId, Double> entry : scores.entrySet()) {
            double zScore = (entry.getValue() - mean) / stdDev;
            
            // Convert Z-score to [0,1] range (approximately)
            // Using cumulative distribution function approximation
            double normalizedScore = 0.5 * (1 + Math.tanh(zScore * 0.5));
            
            normalized.put(entry.getKey(), normalizedScore);
        }
        
        return normalized;
    }
    
    /**
     * Apply sigmoid normalization
     * 
     * @param scores Map of node IDs to raw scores
     * @return Map of node IDs to normalized scores
     */
    private Map<NodeId, Double> applySigmoidNormalization(Map<NodeId, Double> scores) {
        Map<NodeId, Double> normalized = new HashMap<>();
        
        // Apply sigmoid function: 1 / (1 + e^(-x))
        for (Map.Entry<NodeId, Double> entry : scores.entrySet()) {
            double sigmoid = 1.0 / (1.0 + Math.exp(-entry.getValue()));
            normalized.put(entry.getKey(), sigmoid);
        }
        
        return normalized;
    }
    
    /**
     * Apply percentile-based normalization
     * 
     * @param scores Map of node IDs to raw scores
     * @param scoreValues List of score values
     * @return Map of node IDs to normalized scores
     */
    private Map<NodeId, Double> applyPercentileNormalization(Map<NodeId, Double> scores, 
                                                          List<Double> scoreValues) {
        Map<NodeId, Double> normalized = new HashMap<>();
        
        // Sort scores
        List<Double> sortedScores = new ArrayList<>(scoreValues);
        java.util.Collections.sort(sortedScores);
        
        // Calculate percentile for each score
        for (Map.Entry<NodeId, Double> entry : scores.entrySet()) {
            double score = entry.getValue();
            int rank = 0;
            
            // Find rank of this score
            for (Double sortedScore : sortedScores) {
                if (sortedScore < score) {
                    rank++;
                }
            }
            
            // Calculate percentile (0-1 range)
            double percentile = (double) rank / sortedScores.size();
            normalized.put(entry.getKey(), percentile);
        }
        
        return normalized;
    }
    
    /**
     * Set the normalization strategy
     * 
     * @param strategy New normalization strategy
     */
    public void setNormalizationStrategy(NormalizationStrategy strategy) {
        this.currentStrategy = strategy;
        LOG.info("Changed normalization strategy to: {}", strategy);
    }
    
    /**
     * Get the current normalization strategy
     * 
     * @return Current normalization strategy
     */
    public NormalizationStrategy getNormalizationStrategy() {
        return currentStrategy;
    }
    
    /**
     * Normalize a single score based on statistics from a set of scores
     * 
     * @param score Score to normalize
     * @param scores Map of existing scores for context
     * @return Normalized score
     */
    public double normalizeSingleScore(double score, Map<NodeId, Double> scores) {
        if (scores == null || scores.isEmpty()) {
            // No context for normalization, return score directly
            return Math.max(0.0, Math.min(1.0, score));
        }
        
        List<Double> scoreValues = new ArrayList<>(scores.values());
        
        switch (currentStrategy) {
            case MIN_MAX:
                // Find min and max
                double min = Double.MAX_VALUE;
                double max = Double.MIN_VALUE;
                
                for (Double s : scoreValues) {
                    min = Math.min(min, s);
                    max = Math.max(max, s);
                }
                
                if (max == min) {
                    return 0.5;
                }
                
                return (score - min) / (max - min);
                
            case Z_SCORE:
                // Calculate mean and standard deviation
                double sum = 0.0;
                for (Double s : scoreValues) {
                    sum += s;
                }
                double mean = sum / scoreValues.size();
                
                double sumSquaredDiff = 0.0;
                for (Double s : scoreValues) {
                    double diff = s - mean;
                    sumSquaredDiff += diff * diff;
                }
                double stdDev = Math.sqrt(sumSquaredDiff / scoreValues.size());
                
                if (stdDev == 0) {
                    return 0.5;
                }
                
                double zScore = (score - mean) / stdDev;
                return 0.5 * (1 + Math.tanh(zScore * 0.5));
                
            case SIGMOID:
                return 1.0 / (1.0 + Math.exp(-score));
                
            case PERCENTILE:
                // Sort scores
                List<Double> sortedScores = new ArrayList<>(scoreValues);
                java.util.Collections.sort(sortedScores);
                
                // Find rank of this score
                int rank = 0;
                for (Double sortedScore : sortedScores) {
                    if (sortedScore < score) {
                        rank++;
                    }
                }
                
                return (double) rank / sortedScores.size();
                
            case NONE:
            default:
                return score;
        }
    }
    
    /**
     * Apply multi-strategy normalization (average of multiple strategies)
     * 
     * @param scores Map of node IDs to raw scores
     * @return Map of node IDs to normalized scores
     */
    public Map<NodeId, Double> applyMultiStrategyNormalization(Map<NodeId, Double> scores) {
        // Save current strategy
        NormalizationStrategy savedStrategy = currentStrategy;
        
        // Apply each strategy and combine results
        Map<NodeId, Double> result = new HashMap<>();
        
        // Initialize result with zeros
        for (NodeId nodeId : scores.keySet()) {
            result.put(nodeId, 0.0);
        }
        
        // Apply each strategy and add to result
        for (NormalizationStrategy strategy : NormalizationStrategy.values()) {
            if (strategy == NormalizationStrategy.NONE) {
                continue; // Skip NONE strategy
            }
            
            // Set strategy and normalize
            setNormalizationStrategy(strategy);
            Map<NodeId, Double> normalizedScores = normalizeScores(scores);
            
            // Add to result
            for (Map.Entry<NodeId, Double> entry : normalizedScores.entrySet()) {
                NodeId nodeId = entry.getKey();
                double currentValue = result.getOrDefault(nodeId, 0.0);
                result.put(nodeId, currentValue + entry.getValue());
            }
        }
        
        // Calculate average
        int strategyCount = NormalizationStrategy.values().length - 1; // Excluding NONE
        for (NodeId nodeId : result.keySet()) {
            double sum = result.get(nodeId);
            result.put(nodeId, sum / strategyCount);
        }
        
        // Restore original strategy
        setNormalizationStrategy(savedStrategy);
        
        return result;
    }
    
    /**
     * Get normalization statistics for a set of scores
     * 
     * @param scores Map of node IDs to scores
     * @return Map of statistics
     */
    public Map<String, Object> getNormalizationStatistics(Map<NodeId, Double> scores) {
        Map<String, Object> stats = new HashMap<>();
        
        if (scores == null || scores.isEmpty()) {
            return stats;
        }
        
        List<Double> scoreValues = new ArrayList<>(scores.values());
        
        // Calculate basic statistics
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        double sum = 0.0;
        
        for (Double score : scoreValues) {
            min = Math.min(min, score);
            max = Math.max(max, score);
            sum += score;
        }
        
        double mean = sum / scoreValues.size();
        
        double sumSquaredDiff = 0.0;
        for (Double score : scoreValues) {
            double diff = score - mean;
            sumSquaredDiff += diff * diff;
        }
        double stdDev = Math.sqrt(sumSquaredDiff / scoreValues.size());
        
        // Sort for percentiles
        java.util.Collections.sort(scoreValues);
        double median = scoreValues.get(scoreValues.size() / 2);
        
        // Store statistics
        stats.put("count", scoreValues.size());
        stats.put("min", min);
        stats.put("max", max);
        stats.put("mean", mean);
        stats.put("median", median);
        stats.put("stdDev", stdDev);
        stats.put("range", max - min);
        stats.put("strategy", currentStrategy);
        
        return stats;
    }
}