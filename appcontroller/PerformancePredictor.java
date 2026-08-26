package com.perphproctor.appcontroller;

import com.perphproctor.common.Constants;
import com.perphproctor.common.Metrics;
import com.perphproctor.common.Utils;
import com.perphproctor.common.WorkloadTypes.BatchJobType;
import com.perphproctor.common.WorkloadTypes.LraComponentType;
import com.perphproctor.common.WorkloadTypes;
import com.perphproctor.common.WorkloadProfile;
import com.perphproctor.continuouslearning.ContinualLearningRandomForest;
import com.perphproctor.continuouslearning.ModelEvaluator;

import org.apache.hadoop.yarn.api.records.NodeId;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * PerformancePredictor executes interference prediction for long-running applications
 * based on the continuous learning model. It quantifies the impact of co-location on
 * LRA performance for various batch job types and resource configurations.
 * 
 * This class implements equation 2 from the paper, calculating aggregated performance
 * predictions for all co-located LRA components on a node.
 */
public class PerformancePredictor implements Serializable {
    
    private static final long serialVersionUID = 1L;
    private static final Logger LOG = LoggerFactory.getLogger(PerformancePredictor.class);
    
    // Node identification
    private final NodeId nodeId;
    private final String hostname;
    
    // Reference to model manager
    private final ModelManager modelManager;
    
    // Component tracking
    private final Map<String, LraComponentInfo> lraComponents;
    
    // Baseline latencies for services
    private final Map<String, Double> baselineLatencies;
    
    // Prediction cache
    private final Map<BatchJobType, Map<String, Double>> predictionCache;
    private long lastPredictionUpdateTime;
    private static final long PREDICTION_CACHE_VALIDITY_MS = 5000; // 5 seconds
    
    // Thread safety
    private final AtomicBoolean initialized;
    private final ReentrantReadWriteLock lock;
    
    /**
     * Constructor
     * 
     * @param nodeId Node identifier
     * @param modelManager Model manager reference
     */
    public PerformancePredictor(NodeId nodeId, ModelManager modelManager) {
        this.nodeId = nodeId;
        this.hostname = Utils.getHostname();
        this.modelManager = modelManager;
        this.lraComponents = new ConcurrentHashMap<>();
        this.baselineLatencies = new ConcurrentHashMap<>();
        this.predictionCache = new ConcurrentHashMap<>();
        this.initialized = new AtomicBoolean(false);
        this.lock = new ReentrantReadWriteLock();
        this.lastPredictionUpdateTime = 0;
        
        LOG.info("PerformancePredictor initialized for node {}", nodeId);
    }
    
    /**
     * Initialize the predictor
     * 
     * @return true if initialization was successful
     */
    public boolean initialize() {
        if (initialized.compareAndSet(false, true)) {
            LOG.info("Initializing performance predictor for node {}", nodeId);
            
            // Ensure model is initialized
            boolean modelReady = modelManager.initialize();
            if (!modelReady) {
                LOG.error("Failed to initialize model manager");
                initialized.set(false);
                return false;
            }
            
            LOG.info("Performance predictor initialized successfully for node {}", nodeId);
            return true;
        }
        
        LOG.info("Performance predictor already initialized for node {}", nodeId);
        return true;
    }
    
    /**
     * Register an LRA component
     * 
     * @param componentId Component identifier
     * @param serviceName Service name
     * @param componentType Component type
     */
    public void registerComponent(String componentId, String serviceName, LraComponentType componentType) {
        if (componentId == null || serviceName == null || componentType == null) {
            LOG.warn("Cannot register component: Invalid parameters");
            return;
        }
        
        lock.writeLock().lock();
        try {
            LraComponentInfo component = new LraComponentInfo(componentId, serviceName, componentType);
            lraComponents.put(componentId, component);
            
            // Clear prediction cache
            predictionCache.clear();
            
            LOG.info("Registered LRA component: id={}, service={}, type={}",
                    componentId, serviceName, componentType);
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Unregister an LRA component
     * 
     * @param componentId Component identifier
     */
    public void unregisterComponent(String componentId) {
        if (componentId == null) {
            return;
        }
        
        lock.writeLock().lock();
        try {
            LraComponentInfo removed = lraComponents.remove(componentId);
            if (removed != null) {
                // Clear prediction cache
                predictionCache.clear();
                
                LOG.info("Unregistered LRA component: id={}, service={}",
                        componentId, removed.getServiceName());
            }
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Update component metrics
     * 
     * @param componentId Component identifier
     * @param metrics New metrics
     */
    public void updateComponentMetrics(String componentId, Metrics metrics) {
        if (componentId == null || metrics == null) {
            return;
        }
        
        LraComponentInfo component = lraComponents.get(componentId);
        if (component != null) {
            component.updateMetrics(metrics);
            
            // Clear prediction cache when metrics change significantly
            if (hasMajorMetricsChange(component.getLastMetrics(), metrics)) {
                predictionCache.clear();
            }
            
            component.setLastMetrics(metrics);
        }
    }
    
    /**
     * Register a baseline latency for a service
     * 
     * @param serviceName Service name
     * @param baselineLatency Baseline latency value
     */
    public void registerBaselineLatency(String serviceName, double baselineLatency) {
        if (serviceName == null || baselineLatency <= 0) {
            return;
        }
        
        lock.writeLock().lock();
        try {
            baselineLatencies.put(serviceName, baselineLatency);
            LOG.info("Registered baseline latency for service {}: {}", 
                    serviceName, baselineLatency);
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Get the baseline latency for a service
     * 
     * @param serviceName Service name
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
     * Predict node performance for a batch job type (Equation 2)
     * 
     * @param batchType Batch job type to predict for
     * @param nodeMetrics Current node metrics
     * @return Aggregated performance prediction
     */
    public double predictNodePerformance(BatchJobType batchType, Metrics nodeMetrics) {
        if (!initialized.get()) {
            LOG.warn("Cannot predict: Performance predictor not initialized");
            return 0.0;
        }
        
        if (batchType == null || nodeMetrics == null) {
            LOG.warn("Cannot predict: Invalid parameters");
            return 0.0;
        }
        
        lock.readLock().lock();
        try {
            // Check if we have components to predict for
            if (lraComponents.isEmpty()) {
                LOG.debug("No LRA components registered, returning default prediction");
                return 1.0; // Optimal performance when no LRAs
            }
            
            // Calculate predicted latency for each component
            double totalPredictedLatency = 0.0;
            int componentCount = 0;
            
            for (LraComponentInfo component : lraComponents.values()) {
                // Get baseline latency for this service
                String serviceName = component.getServiceName();
                Double baseline = baselineLatencies.get(serviceName);
                
                if (baseline == null || baseline <= 0) {
                    LOG.debug("No baseline latency for service {}, skipping", serviceName);
                    continue;
                }
                
                // Create combined metrics
                Metrics combinedMetrics = new Metrics();
                combinedMetrics.merge(nodeMetrics);
                combinedMetrics.merge(component.getMetrics());
                
                // Set batch type in metrics
                combinedMetrics.setBatchType(batchType.getCode());
                
                // Predict latency
                double predictedLatency = modelManager.predict(combinedMetrics);
                totalPredictedLatency += predictedLatency;
                componentCount++;
                
                LOG.debug("Predicted latency for component {}: {} (baseline: {})",
                        component.getComponentId(), predictedLatency, baseline);
            }
            
            // Calculate average prediction
            if (componentCount > 0) {
                return totalPredictedLatency / componentCount;
            } else {
                return 0.0;
            }
        } catch (Exception e) {
            LOG.error("Error predicting performance: {}", e.getMessage(), e);
            return 0.0;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Predict performance for all services with a specific batch job type
     * 
     * @param batchType Batch job type
     * @param nodeMetrics Current node metrics
     * @return Map of service names to predicted latencies
     */
    public Map<String, Double> predictServicePerformance(BatchJobType batchType, Metrics nodeMetrics) {
        if (!initialized.get()) {
            LOG.warn("Cannot predict: Performance predictor not initialized");
            return new HashMap<>();
        }
        
        // Check if we can use cached predictions
        if (isCacheValid() && predictionCache.containsKey(batchType)) {
            return new HashMap<>(predictionCache.get(batchType));
        }
        
        lock.readLock().lock();
        try {
            Map<String, Double> predictions = new HashMap<>();
            
            // Predict for each component
            for (LraComponentInfo component : lraComponents.values()) {
                String serviceName = component.getServiceName();
                
                // Create combined metrics
                Metrics combinedMetrics = new Metrics();
                combinedMetrics.merge(nodeMetrics);
                combinedMetrics.merge(component.getMetrics());
                
                // Set batch type in metrics
                combinedMetrics.setBatchType(batchType.getCode());
                
                // Predict latency
                double predictedLatency = modelManager.predict(combinedMetrics);
                
                // LAIP: scale by the layer's dominant contended resource
                predictedLatency *= layerSensitivityFactor(component.getComponentType(), nodeMetrics);

                // Add to predictions
                predictions.put(serviceName, predictedLatency);
            }
            
            // Update cache
            predictionCache.put(batchType, new HashMap<>(predictions));
            lastPredictionUpdateTime = System.currentTimeMillis();
            
            return predictions;
        } catch (Exception e) {
            LOG.error("Error predicting service performance: {}", e.getMessage(), e);
            return new HashMap<>();
        } finally {
            lock.readLock().unlock();
        }
    }
    
    // Picks the resource with highest (sensitivity x utilization) for this layer
    // and scales the prediction by its severity.
    private double layerSensitivityFactor(LraComponentType type, Metrics nodeMetrics) {
        if (type == null) return 1.0;

        WorkloadProfile p = WorkloadTypes.createLraProfile("laip-ref", type.name(), type);

        double cpuU = nodeMetrics.getCpuUtilization();
        double memU = nodeMetrics.getMemoryUtilization();
        double llcU = nodeMetrics.getLlcUtilization();
        double mbwU = nodeMetrics.getMemoryBandwidth() / 1000.0; // see ResourcePatternDetector
        double ioU  = nodeMetrics.getIoThroughput()  / 5.0;

        final double MAX_SENS = 10.0;
        double[] sens = { p.getCpuSensitivity(), p.getMemorySensitivity(), p.getLlcSensitivity(),
                           p.getMbwSensitivity(), p.getIoSensitivity() };
        double[] util = { cpuU, memU, llcU, mbwU, ioU };

        int dominant = 0;
        double maxScore = -1.0;
        for (int i = 0; i < sens.length; i++) {
            double score = sens[i] * util[i];
            if (score > maxScore) { maxScore = score; dominant = i; }
        }

        // ALPHA caps the factor at 1+ALPHA=1.5, matching the max tail-latency multiplier observed
        // in our deployment data. Not the original production constant (unavailable); placeholder.
        final double ALPHA = 0.5;
        return 1.0 + (sens[dominant] / MAX_SENS) * (util[dominant] / 100.0) * ALPHA;
    }

    /**
     * Calculate normalized performance impact based on baseline latencies
     * 
     * @param batchType Batch job type
     * @param nodeMetrics Current node metrics
     * @return Normalized performance impact ratio (0-1, lower is better)
     */
    public double calculatePerformanceImpact(BatchJobType batchType, Metrics nodeMetrics) {
        if (!initialized.get()) {
            LOG.warn("Cannot calculate impact: Performance predictor not initialized");
            return 0.0;
        }
        
        lock.readLock().lock();
        try {
            // Get predictions for all services
            Map<String, Double> predictions = predictServicePerformance(batchType, nodeMetrics);
            
            double totalPredicted = 0.0;
            double totalBaseline = 0.0;
            int serviceCount = 0;
            
            // Compare with baselines
            for (Map.Entry<String, Double> entry : predictions.entrySet()) {
                String serviceName = entry.getKey();
                Double predicted = entry.getValue();
                Double baseline = baselineLatencies.get(serviceName);
                
                if (baseline != null && baseline > 0) {
                    totalPredicted += predicted;
                    totalBaseline += baseline;
                    serviceCount++;
                }
            }
            
            if (serviceCount > 0) {
                // Calculate average impact ratio
                double ratio = (totalPredicted - totalBaseline) / totalBaseline;
                
                // Ensure ratio is non-negative
                return Math.max(0.0, ratio);
            } else {
                return 0.0;
            }
        } catch (Exception e) {
            LOG.error("Error calculating performance impact: {}", e.getMessage(), e);
            return 0.0;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Check if node should accept a task of given batch type
     * 
     * @param batchType Batch job type
     * @param nodeMetrics Current node metrics
     * @param threshold Acceptance threshold
     * @return true if node should accept task
     */
    public boolean shouldAcceptTask(BatchJobType batchType, Metrics nodeMetrics, double threshold) {
        if (!initialized.get()) {
            LOG.warn("Cannot determine acceptance: Performance predictor not initialized");
            return false;
        }
        
        // Calculate performance impact
        double impact = calculatePerformanceImpact(batchType, nodeMetrics);
        
        // Compare with threshold
        return impact <= threshold;
    }
    
    /**
     * Get prediction statistics
     * 
     * @return Map of statistics
     */
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new HashMap<>();
        
        // Add component statistics
        stats.put("componentCount", lraComponents.size());
        stats.put("baselineLatencyCount", baselineLatencies.size());
        
        // Add prediction statistics
        stats.put("cacheEntryCount", predictionCache.size());
        stats.put("lastPredictionTime", lastPredictionUpdateTime);
        stats.put("predictionCacheValid", isCacheValid());
        
        // Add model statistics
        stats.put("modelStats", modelManager.getStatistics());
        
        return stats;
    }
    
    /**
     * Check if prediction cache is valid
     * 
     * @return true if cache is valid
     */
    private boolean isCacheValid() {
        long currentTime = System.currentTimeMillis();
        return (currentTime - lastPredictionUpdateTime) < PREDICTION_CACHE_VALIDITY_MS;
    }
    
    /**
     * Check if metrics have changed significantly
     * 
     * @param oldMetrics Previous metrics
     * @param newMetrics New metrics
     * @return true if significant change detected
     */
    private boolean hasMajorMetricsChange(Metrics oldMetrics, Metrics newMetrics) {
        if (oldMetrics == null || newMetrics == null) {
            return true;
        }
        
        // Check if CPU utilization changed by more than 20%
        if (Math.abs(oldMetrics.getCpuUtilization() - newMetrics.getCpuUtilization()) > 20.0) {
            return true;
        }
        
        // Check if memory utilization changed by more than 15%
        if (Math.abs(oldMetrics.getMemoryUtilization() - newMetrics.getMemoryUtilization()) > 15.0) {
            return true;
        }
        
        // Check if I/O throughput changed by more than 30%
        if (Math.abs(oldMetrics.getIoThroughput() - newMetrics.getIoThroughput()) > 30.0) {
            return true;
        }
        
        return false;
    }
    
    /**
     * Inner class for LRA component information
     */
    private static class LraComponentInfo implements Serializable {
        private static final long serialVersionUID = 1L;
        
        private final String componentId;
        private final String serviceName;
        private final LraComponentType componentType;
        private final Metrics metrics;
        private Metrics lastMetrics;
        
        public LraComponentInfo(String componentId, String serviceName, LraComponentType componentType) {
            this.componentId = componentId;
            this.serviceName = serviceName;
            this.componentType = componentType;
            this.metrics = new Metrics();
            this.lastMetrics = null;
        }
        
        public String getComponentId() {
            return componentId;
        }
        
        public String getServiceName() {
            return serviceName;
        }
        
        public LraComponentType getComponentType() {
            return componentType;
        }
        
        public Metrics getMetrics() {
            return metrics;
        }
        
        public Metrics getLastMetrics() {
            return lastMetrics;
        }
        
        public void setLastMetrics(Metrics metrics) {
            this.lastMetrics = metrics;
        }
        
        public void updateMetrics(Metrics newMetrics) {
            if (newMetrics != null) {
                metrics.merge(newMetrics);
            }
        }
    }
}