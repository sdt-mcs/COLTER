package com.perphproctor.recommendation;

import com.perphproctor.common.Constants;
import com.perphproctor.common.Metrics;
import com.perphproctor.common.Utils;
import com.perphproctor.common.WorkloadTypes;
import com.perphproctor.continuouslearning.ContinualLearningRandomForest;
import org.apache.hadoop.yarn.api.records.NodeId;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * RecommendationCalculator computes node recommendation scores based on
 * performance prediction of long-running applications (LRAs).
 * 
 * This class implements Equation 3 from the paper, calculating recommendation
 * scores for scheduling decisions based on predicted LRA performance.
 */
public class RecommendationCalculator {
    
    private static final Logger LOG = LoggerFactory.getLogger(RecommendationCalculator.class);
    
    // Node recommendation scores
    private final Map<NodeId, Double> nodeRecommendationScores;
    
    // Performance model
    private final ContinualLearningRandomForest predictionModel;
    
    // Baseline (minimum) latency for each LRA service
    private final Map<String, Double> baselineLatencies;
    
    // Reference to confidence checker
    private final ConfidenceChecker confidenceChecker;
    
    // Node component mapping
    private final Map<NodeId, List<LraComponent>> nodeComponentMap;
    
    // Sensitivity parameter k (equation 3)
    private final double sensitivityParameter;
    
    // Thread safety
    private final ReentrantReadWriteLock lock;
    
    /**
     * Constructor with default sensitivity parameter
     * 
     * @param predictionModel Performance prediction model
     */
    public RecommendationCalculator(ContinualLearningRandomForest predictionModel) {
        this(predictionModel, Constants.DEFAULT_SENSITIVITY_PARAMETER_K);
    }
    
    /**
     * Constructor with custom sensitivity parameter
     * 
     * @param predictionModel Performance prediction model
     * @param sensitivityParameter Sensitivity parameter k
     */
    public RecommendationCalculator(ContinualLearningRandomForest predictionModel, 
                                    double sensitivityParameter) {
        this.predictionModel = predictionModel;
        this.sensitivityParameter = sensitivityParameter;
        
        this.nodeRecommendationScores = new ConcurrentHashMap<>();
        this.baselineLatencies = new ConcurrentHashMap<>();
        this.confidenceChecker = new ConfidenceChecker(this);
        this.nodeComponentMap = new ConcurrentHashMap<>();
        this.lock = new ReentrantReadWriteLock();
        
        LOG.info("RecommendationCalculator initialized with sensitivity parameter k={}", 
                sensitivityParameter);
    }
    
    /**
     * Register a baseline (minimum) latency for an LRA service
     * 
     * @param serviceName LRA service name
     * @param baselineLatency Baseline latency value
     */
    public void registerBaselineLatency(String serviceName, double baselineLatency) {
        if (serviceName != null && baselineLatency > 0) {
            lock.writeLock().lock();
            try {
                baselineLatencies.put(serviceName, baselineLatency);
                LOG.info("Registered baseline latency for service {}: {}", 
                        serviceName, baselineLatency);
            } finally {
                lock.writeLock().unlock();
            }
        }
    }
    
    /**
     * Get the baseline latency for an LRA service
     * 
     * @param serviceName LRA service name
     * @return Baseline latency or null if not registered
     */
    public Double getBaselineLatency(String serviceName) {
        if (serviceName == null) {
            return null;
        }
        
        lock.readLock().lock();
        try {
            return baselineLatencies.get(serviceName);
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Register an LRA component on a node
     * 
     * @param nodeId Node identifier
     * @param component LRA component information
     */
    public void registerNodeComponent(NodeId nodeId, LraComponent component) {
        if (nodeId == null || component == null) {
            return;
        }
        
        lock.writeLock().lock();
        try {
            List<LraComponent> components = nodeComponentMap.computeIfAbsent(
                    nodeId, k -> new ArrayList<>());
            
            // Remove existing component with same ID if present
            components.removeIf(c -> c.getComponentId().equals(component.getComponentId()));
            
            // Add new component
            components.add(component);
            
            LOG.info("Registered component {} on node {}", 
                    component.getComponentId(), nodeId);
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Remove an LRA component from a node
     * 
     * @param nodeId Node identifier
     * @param componentId Component identifier
     */
    public void removeNodeComponent(NodeId nodeId, String componentId) {
        if (nodeId == null || componentId == null) {
            return;
        }
        
        lock.writeLock().lock();
        try {
            List<LraComponent> components = nodeComponentMap.get(nodeId);
            if (components != null) {
                components.removeIf(c -> c.getComponentId().equals(componentId));
                
                if (components.isEmpty()) {
                    nodeComponentMap.remove(nodeId);
                }
                
                LOG.info("Removed component {} from node {}", componentId, nodeId);
            }
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Calculate recommendation score for a node based on equation 3
     * 
     * @param nodeId Node identifier
     * @param metrics Current node metrics
     * @param batchType Type of batch job being scheduled
     * @return Recommendation score (0-1)
     */
    public double calculateRecommendationScore(NodeId nodeId, Metrics metrics, 
                                             WorkloadTypes.BatchJobType batchType) {
        if (nodeId == null || metrics == null) {
            return 0.0;
        }
        
        lock.readLock().lock();
        try {
            List<LraComponent> components = nodeComponentMap.getOrDefault(nodeId, 
                                                                       Collections.emptyList());
            
            // If no LRA components on node, return maximum score
            if (components.isEmpty()) {
                return 1.0;
            }
            
            // Calculate latency prediction for each component
            double totalPredictedLatency = 0.0;
            double totalBaselineLatency = 0.0;
            int serviceCount = 0;
            
            for (LraComponent component : components) {
                // Get baseline latency for this service
                String serviceName = component.getServiceName();
                Double baseline = baselineLatencies.get(serviceName);
                
                if (baseline == null || baseline <= 0) {
                    // Skip components with unknown baseline
                    continue;
                }
                
                // Set batch type in metrics
                metrics.setBatchType(batchType.getCode());
                
                // Prepare feature vector for prediction
                // Include component-specific metrics
                Metrics componentMetrics = new Metrics();
                componentMetrics.merge(metrics);
                componentMetrics.merge(component.getComponentMetrics());
                
                // Predict latency for this component
                double[] features = componentMetrics.toFeatureVector();
                double predictedLatency = predictionModel.predict(features);
                
                totalPredictedLatency += predictedLatency;
                totalBaselineLatency += baseline;
                serviceCount++;
            }
            
            // If no valid services found, return default score
            if (serviceCount == 0) {
                return 0.5;
            }
            
            // Calculate latency growth ratio
            double latencyGrowthRatio = (totalPredictedLatency - totalBaselineLatency) / totalBaselineLatency;
            
            // Apply equation 3: Cn = 1 / exp(latencyGrowthRatio * k)
            double score = 1.0 / Math.exp(latencyGrowthRatio * sensitivityParameter);
            
            // Ensure score is within valid range
            score = Math.max(0.0, Math.min(1.0, score));
            
            // Update node recommendation score
            nodeRecommendationScores.put(nodeId, score);
            
            LOG.debug("Calculated recommendation score for node {}: {} (baseline={}, predicted={}, ratio={})",
                    nodeId, score, totalBaselineLatency, totalPredictedLatency, latencyGrowthRatio);
            
            return score;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Calculate recommendation scores for all nodes
     * 
     * @param nodesMetrics Map of node metrics
     * @param batchType Type of batch job being scheduled
     * @return Map of node IDs to recommendation scores
     */
    public Map<NodeId, Double> calculateAllRecommendationScores(
            Map<NodeId, Metrics> nodesMetrics, WorkloadTypes.BatchJobType batchType) {
        
        Map<NodeId, Double> scores = new HashMap<>();
        
        for (Map.Entry<NodeId, Metrics> entry : nodesMetrics.entrySet()) {
            NodeId nodeId = entry.getKey();
            Metrics metrics = entry.getValue();
            
            double score = calculateRecommendationScore(nodeId, metrics, batchType);
            scores.put(nodeId, score);
        }
        
        return scores;
    }
    
    /**
     * Calculate multiple service predictions for a node
     * 
     * @param nodeId Node identifier
     * @param metrics Current node metrics
     * @param batchType Type of batch job being scheduled
     * @return Map of service names to predicted latencies
     */
    public Map<String, Double> calculateServicePredictions(NodeId nodeId, Metrics metrics,
                                                        WorkloadTypes.BatchJobType batchType) {
        if (nodeId == null || metrics == null) {
            return Collections.emptyMap();
        }
        
        lock.readLock().lock();
        try {
            List<LraComponent> components = nodeComponentMap.getOrDefault(nodeId, 
                                                                       Collections.emptyList());
            
            // If no LRA components on node, return empty map
            if (components.isEmpty()) {
                return Collections.emptyMap();
            }
            
            Map<String, Double> predictions = new HashMap<>();
            
            // Calculate latency prediction for each component
            for (LraComponent component : components) {
                // Set batch type in metrics
                metrics.setBatchType(batchType.getCode());
                
                // Prepare feature vector for prediction
                // Include component-specific metrics
                Metrics componentMetrics = new Metrics();
                componentMetrics.merge(metrics);
                componentMetrics.merge(component.getComponentMetrics());
                
                // Predict latency for this component
                double[] features = componentMetrics.toFeatureVector();
                double predictedLatency = predictionModel.predict(features);
                
                // Store prediction
                predictions.put(component.getServiceName(), predictedLatency);
            }
            
            return predictions;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Get all current node recommendation scores
     * 
     * @return Map of node IDs to recommendation scores
     */
    public Map<NodeId, Double> getNodeRecommendationScores() {
        return new HashMap<>(nodeRecommendationScores);
    }
    
    /**
     * Get the recommendation score for a specific node
     * 
     * @param nodeId Node identifier
     * @return Recommendation score, or 0.5 if not available
     */
    public double getNodeRecommendationScore(NodeId nodeId) {
        if (nodeId == null) {
            return 0.5;
        }
        
        return nodeRecommendationScores.getOrDefault(nodeId, 0.5);
    }
    
    /**
     * Get the confidence checker
     * 
     * @return Confidence checker
     */
    public ConfidenceChecker getConfidenceChecker() {
        return confidenceChecker;
    }
    
    /**
     * Get statistics about recommendation scores
     * 
     * @return Map of statistics
     */
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new HashMap<>();
        
        // Get all scores
        List<Double> scores = new ArrayList<>(nodeRecommendationScores.values());
        
        // Calculate statistics
        double mean = Utils.mean(scores);
        double stdDev = Utils.standardDeviation(scores);
        double p95 = Utils.percentile95(scores);
        
        stats.put("scoreCount", scores.size());
        stats.put("scoreMean", mean);
        stats.put("scoreStdDev", stdDev);
        stats.put("scoreP95", p95);
        stats.put("sensitivityParameter", sensitivityParameter);
        stats.put("baselineLatencyCount", baselineLatencies.size());
        
        return stats;
    }
    
    /**
     * Class representing an LRA component on a node
     */
    public static class LraComponent {
        private final String componentId;
        private final String serviceName;
        private final WorkloadTypes.LraComponentType componentType;
        private final Metrics componentMetrics;
        
        /**
         * Constructor
         * 
         * @param componentId Unique component identifier
         * @param serviceName Service name
         * @param componentType Component type
         */
        public LraComponent(String componentId, String serviceName, 
                          WorkloadTypes.LraComponentType componentType) {
            this.componentId = componentId;
            this.serviceName = serviceName;
            this.componentType = componentType;
            this.componentMetrics = new Metrics();
        }
        
        /**
         * Get the component identifier
         * 
         * @return Component identifier
         */
        public String getComponentId() {
            return componentId;
        }
        
        /**
         * Get the service name
         * 
         * @return Service name
         */
        public String getServiceName() {
            return serviceName;
        }
        
        /**
         * Get the component type
         * 
         * @return Component type
         */
        public WorkloadTypes.LraComponentType getComponentType() {
            return componentType;
        }
        
        /**
         * Get the component-specific metrics
         * 
         * @return Component metrics
         */
        public Metrics getComponentMetrics() {
            return componentMetrics;
        }
        
        /**
         * Update component metrics
         * 
         * @param metrics New metrics
         */
        public void updateMetrics(Metrics metrics) {
            if (metrics != null) {
                componentMetrics.merge(metrics);
            }
        }
        
        @Override
        public String toString() {
            return "LraComponent{" +
                   "componentId='" + componentId + '\'' +
                   ", serviceName='" + serviceName + '\'' +
                   ", componentType=" + componentType +
                   '}';
        }
    }
}