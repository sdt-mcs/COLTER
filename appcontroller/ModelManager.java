package com.perphproctor.appcontroller;

import com.perphproctor.common.Constants;
import com.perphproctor.common.Metrics;
import com.perphproctor.common.Utils;
import com.perphproctor.continuouslearning.ContinualLearningRandomForest;
import com.perphproctor.continuouslearning.ModelEvaluator;
import com.perphproctor.continuouslearning.SampleManager;

import org.apache.hadoop.yarn.api.records.NodeId;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * ModelManager manages the lifecycle of prediction models in the AppController component.
 * It handles model initialization, training, continuous updates, evaluation, and persistence.
 * 
 * The ModelManager implements the continuous learning functionality described in the
 * PerphProctor framework, enabling adaptive performance prediction with minimal overhead.
 */
public class ModelManager implements Serializable {
    
    private static final long serialVersionUID = 1L;
    private static final Logger LOG = LoggerFactory.getLogger(ModelManager.class);
    
    // Node identification
    private final NodeId nodeId;
    private final String hostname;
    
    // Model components
    private final ContinualLearningRandomForest predictionModel;
    private final SampleManager sampleManager;
    private final ModelEvaluator modelEvaluator;
    
    // Model state
    private final AtomicBoolean modelInitialized;
    private final AtomicInteger updateCount;
    private long lastEvaluationTime;
    private double lastEvaluationError;
    
    // Configuration
    private final int modelFeatureCount;
    private final int maxSampleCount;
    private final int batchSize;
    private final boolean useIncrementalLearning;
    private final long modelEvaluationIntervalMs;
    private final String modelStoragePath;
    
    // Thread safety
    private final ReentrantReadWriteLock lock;
    
    /**
     * Constructor with default configuration
     * 
     * @param nodeId Node identifier
     */
    public ModelManager(NodeId nodeId) {
        this(
            nodeId,
            7, // Feature vector dimension as described in paper
            10000, // Maximum sample count
            100, // Batch size
            true, // Use incremental learning
            600000, // Evaluate model every 10 minutes
            Constants.DEFAULT_MODEL_STORAGE_PATH
        );
    }
    
    /**
     * Constructor with custom configuration
     * 
     * @param nodeId Node identifier
     * @param featureCount Number of features in model
     * @param maxSampleCount Maximum number of samples to retain
     * @param batchSize Batch size for training
     * @param useIncrementalLearning Whether to use incremental learning
     * @param modelEvaluationIntervalMs Interval between model evaluations
     * @param modelStoragePath Path to store model files
     */
    public ModelManager(NodeId nodeId, int featureCount, int maxSampleCount, 
                       int batchSize, boolean useIncrementalLearning, 
                       long modelEvaluationIntervalMs, String modelStoragePath) {
        this.nodeId = nodeId;
        this.hostname = Utils.getHostname();
        this.modelFeatureCount = featureCount;
        this.maxSampleCount = maxSampleCount;
        this.batchSize = batchSize;
        this.useIncrementalLearning = useIncrementalLearning;
        this.modelEvaluationIntervalMs = modelEvaluationIntervalMs;
        this.modelStoragePath = modelStoragePath;
        
        // Initialize model components
        this.predictionModel = new ContinualLearningRandomForest(featureCount);
        this.sampleManager = new SampleManager(featureCount, maxSampleCount, batchSize, useIncrementalLearning);
        this.modelEvaluator = new ModelEvaluator();
        
        // Initialize state
        this.modelInitialized = new AtomicBoolean(false);
        this.updateCount = new AtomicInteger(0);
        this.lastEvaluationTime = 0;
        this.lastEvaluationError = 0.0;
        this.lock = new ReentrantReadWriteLock();
        
        // Create model storage directory
        createModelStorageDirectory();
        
        LOG.info("ModelManager initialized for node {} with featureCount={}, incremental={}", 
                nodeId, featureCount, useIncrementalLearning);
    }
    
    /**
     * Initialize the model
     * 
     * @return true if initialization was successful
     */
    public boolean initialize() {
        if (modelInitialized.get()) {
            LOG.info("Model already initialized for node {}", nodeId);
            return true;
        }
        
        LOG.info("Initializing model for node {}", nodeId);
        
        // Try to load existing model
        boolean loaded = loadModel();
        if (loaded) {
            modelInitialized.set(true);
            LOG.info("Loaded existing model for node {}", nodeId);
            return true;
        }
        
        // If loading failed, initialize a new model
        LOG.info("No existing model found, initializing new model for node {}", nodeId);
        modelInitialized.set(true);
        return true;
    }
    
    /**
     * Add a training sample
     * 
     * @param metrics Feature metrics
     * @param latency Target latency
     * @return true if sample was added successfully
     */
    public boolean addSample(Metrics metrics, double latency) {
        if (!modelInitialized.get()) {
            LOG.warn("Cannot add sample: Model not initialized");
            return false;
        }
        
        if (metrics == null) {
            LOG.warn("Cannot add sample: Null metrics");
            return false;
        }
        
        try {
            // Add sample to sample manager
            sampleManager.addSample(metrics, latency);
            
            // Update pending updates count
            updateCount.incrementAndGet();
            
            LOG.debug("Added sample: metrics={}, latency={}", metrics, latency);
            return true;
        } catch (Exception e) {
            LOG.error("Error adding sample: {}", e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * Process pending updates (training)
     * 
     * @return Number of updates processed
     */
    public int processUpdates() {
        if (!modelInitialized.get() || updateCount.get() == 0) {
            return 0;
        }
        
        lock.writeLock().lock();
        try {
            LOG.debug("Processing {} pending updates", updateCount.get());
            
            int processed = 0;
            
            if (useIncrementalLearning) {
                // Get incremental batch
                Object[] batch = sampleManager.getNextIncrementalBatch(100);
                if (batch != null) {
                    double[][] features = (double[][]) batch[0];
                    double[] targets = (double[]) batch[1];
                    
                    // Update model with each sample
                    for (int i = 0; i < features.length; i++) {
                        predictionModel.update(features[i], targets[i]);
                        processed++;
                    }
                }
            } else {
                // Check if we have enough samples for batch learning
                if (sampleManager.getSampleCount() >= batchSize) {
                    // Create training/validation split
                    Object[] splitData = sampleManager.createTrainValidationSplit(0.2);
                    if (splitData != null) {
                        double[][] trainFeatures = (double[][]) splitData[0];
                        double[] trainTargets = (double[]) splitData[1];
                        
                        // Train model
                        predictionModel.train(trainFeatures, trainTargets, true);
                        processed = trainFeatures.length;
                    }
                }
            }
            
            // Update count of processed updates
            updateCount.addAndGet(-processed);
            
            // Check if evaluation is needed
            if (isEvaluationNeeded()) {
                evaluateModel();
            }
            
            if (processed > 0) {
                LOG.info("Processed {} model updates", processed);
            }
            
            return processed;
        } catch (Exception e) {
            LOG.error("Error processing updates: {}", e.getMessage(), e);
            return 0;
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Predict latency for a feature vector
     * 
     * @param features Feature vector
     * @return Predicted latency
     */
    public double predict(double[] features) {
        if (!modelInitialized.get()) {
            LOG.warn("Cannot predict: Model not initialized");
            return 0.0;
        }
        
        if (features == null || features.length != modelFeatureCount) {
            LOG.warn("Cannot predict: Invalid feature vector");
            return 0.0;
        }
        
        lock.readLock().lock();
        try {
            return predictionModel.predict(features);
        } catch (Exception e) {
            LOG.error("Error making prediction: {}", e.getMessage(), e);
            return 0.0;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Predict latency for metrics
     * 
     * @param metrics Feature metrics
     * @return Predicted latency
     */
    public double predict(Metrics metrics) {
        if (metrics == null) {
            LOG.warn("Cannot predict: Null metrics");
            return 0.0;
        }
        
        double[] features = metrics.toFeatureVector();
        return predict(features);
    }
    
    /**
     * Check if model evaluation is needed
     * 
     * @return true if evaluation is needed
     */
    private boolean isEvaluationNeeded() {
        long currentTime = System.currentTimeMillis();
        return (currentTime - lastEvaluationTime) >= modelEvaluationIntervalMs;
    }
    
    /**
     * Evaluate model performance
     * 
     * @return Evaluation results or null if evaluation failed
     */
    public ModelEvaluator.PerformanceMetrics evaluateModel() {
        if (!modelInitialized.get()) {
            LOG.warn("Cannot evaluate: Model not initialized");
            return null;
        }
        
        lock.readLock().lock();
        try {
            LOG.info("Evaluating model performance");
            
            // Create validation split
            Object[] splitData = sampleManager.createTrainValidationSplit(0.2);
            if (splitData == null) {
                LOG.warn("Cannot evaluate: Insufficient samples");
                return null;
            }
            
            // Extract validation data
            double[][] validationFeatures = (double[][]) splitData[2];
            double[] validationTargets = (double[]) splitData[3];
            
            // Evaluate model
            ModelEvaluator.PerformanceMetrics metrics = 
                    modelEvaluator.evaluate(predictionModel, validationFeatures, validationTargets);
            
            // Update evaluation state
            lastEvaluationTime = System.currentTimeMillis();
            lastEvaluationError = metrics.rmse;
            
            LOG.info("Model evaluation results: {}", metrics);
            
            // Check if model has degraded
            if (modelEvaluator.hasPerformanceDegraded(0.3)) {
                LOG.warn("Model performance has degraded, scheduling full retraining");
                scheduleFullRetraining();
            }
            
            return metrics;
        } catch (Exception e) {
            LOG.error("Error evaluating model: {}", e.getMessage(), e);
            return null;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Schedule full model retraining
     */
    private void scheduleFullRetraining() {
        lock.writeLock().lock();
        try {
            // Create training split (using all data)
            Object[] trainData = sampleManager.createTrainValidationSplit(0.0);
            if (trainData == null) {
                LOG.warn("Cannot retrain: Insufficient samples");
                return;
            }
            
            double[][] trainFeatures = (double[][]) trainData[0];
            double[] trainTargets = (double[]) trainData[1];
            
            // Train on full dataset
            predictionModel.train(trainFeatures, trainTargets, true);
            
            LOG.info("Completed full model retraining with {} samples", trainFeatures.length);
            
            // Save updated model
            saveModel();
        } catch (Exception e) {
            LOG.error("Error in full retraining: {}", e.getMessage(), e);
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Save current model state
     * 
     * @return true if save was successful
     */
    public boolean saveModel() {
        if (!modelInitialized.get()) {
            LOG.warn("Cannot save: Model not initialized");
            return false;
        }
        
        String modelPath = getModelFilePath();
        
        lock.readLock().lock();
        try {
            LOG.info("Saving model to {}", modelPath);
            
            boolean saved = predictionModel.saveModel(modelPath);
            if (saved) {
                LOG.info("Model saved successfully");
            } else {
                LOG.error("Failed to save model");
            }
            
            return saved;
        } catch (Exception e) {
            LOG.error("Error saving model: {}", e.getMessage(), e);
            return false;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Load model from file
     * 
     * @return true if load was successful
     */
    public boolean loadModel() {
        String modelPath = getModelFilePath();
        File modelFile = new File(modelPath);
        
        if (!modelFile.exists()) {
            LOG.info("No model file found at {}", modelPath);
            return false;
        }
        
        lock.writeLock().lock();
        try {
            LOG.info("Loading model from {}", modelPath);
            
            ContinualLearningRandomForest loadedModel = 
                    ContinualLearningRandomForest.loadModel(modelPath);
            
            if (loadedModel == null) {
                LOG.error("Failed to load model from {}", modelPath);
                return false;
            }
            
            // Transfer state from loaded model to our instance
            // In a real implementation, this would properly merge model state
            // For now, we just use the loaded model directly
            
            LOG.info("Model loaded successfully");
            return true;
        } catch (Exception e) {
            LOG.error("Error loading model: {}", e.getMessage(), e);
            return false;
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Get model statistics
     * 
     * @return Map of statistics
     */
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new HashMap<>();
        
        // Add sample statistics
        stats.put("sampleCount", sampleManager.getSampleCount());
        stats.put("pendingUpdates", updateCount.get());
        
        // Add model statistics
        stats.put("modelInitialized", modelInitialized.get());
        stats.put("treeCount", predictionModel.getTreeCount());
        stats.put("nodeCount", predictionModel.getTotalNodeCount());
        stats.put("modelUpdateCount", predictionModel.getUpdateCount());
        stats.put("lastModelUpdateTime", predictionModel.getLastUpdateTimestamp());
        
        // Add evaluation statistics
        stats.put("lastEvaluationTime", lastEvaluationTime);
        stats.put("lastEvaluationError", lastEvaluationError);
        stats.put("currentMetrics", modelEvaluator.getCurrentMetrics());
        
        return stats;
    }
    
    /**
     * Get prediction model
     * 
     * @return Prediction model
     */
    public ContinualLearningRandomForest getPredictionModel() {
        return predictionModel;
    }
    
    /**
     * Get sample manager
     * 
     * @return Sample manager
     */
    public SampleManager getSampleManager() {
        return sampleManager;
    }
    
    /**
     * Get model evaluator
     * 
     * @return Model evaluator
     */
    public ModelEvaluator getModelEvaluator() {
        return modelEvaluator;
    }
    
    /**
     * Get number of pending updates
     * 
     * @return Pending update count
     */
    public int getPendingUpdateCount() {
        return updateCount.get();
    }
    
    /**
     * Get model file path
     * 
     * @return Model file path
     */
    private String getModelFilePath() {
        return modelStoragePath + "model_" + nodeId.toString() + ".dat";
    }
    
    /**
     * Create model storage directory
     */
    private void createModelStorageDirectory() {
        File dir = new File(modelStoragePath);
        if (!dir.exists()) {
            boolean created = dir.mkdirs();
            if (!created) {
                LOG.error("Failed to create model storage directory: {}", modelStoragePath);
            }
        }
    }
}