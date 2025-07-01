package com.perphproctor.continuouslearning;

import com.perphproctor.common.Constants;
import com.perphproctor.common.Metrics;
import com.perphproctor.common.Utils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Random;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Manages training samples for the continuous learning process.
 * This class handles sample collection, preprocessing, batching,
 * and providing stratified samples for model training and validation.
 */
public class SampleManager implements Serializable {
    
    private static final long serialVersionUID = 1L;
    private static final Random random = new Random();
    
    // Configuration parameters
    private final int maxSampleCount;
    private final int batchSize;
    private final int featureCount;
    private final boolean useIncrementalLearning;
    
    // Feature scaling parameters
    private double[] featureMeans;
    private double[] featureStdDevs;
    
    // Sample storage
    private final List<double[]> featureVectors;
    private final List<Double> targetValues;
    
    // Incremental sample queue for continuous updates
    private transient Queue<TrainingSample> incrementalSamples;
    
    // Thread safety
    private final transient ReentrantReadWriteLock lock;
    private final transient AtomicInteger pendingSamples;
    
    /**
     * Constructor with configuration parameters
     * 
     * @param featureCount Number of features in each sample
     * @param maxSampleCount Maximum number of samples to retain
     * @param batchSize Batch size for training
     * @param useIncrementalLearning Whether to use incremental learning mode
     */
    public SampleManager(int featureCount, int maxSampleCount, int batchSize, boolean useIncrementalLearning) {
        this.featureCount = featureCount;
        this.maxSampleCount = maxSampleCount;
        this.batchSize = batchSize;
        this.useIncrementalLearning = useIncrementalLearning;
        
        this.featureVectors = new ArrayList<>();
        this.targetValues = new ArrayList<>();
        this.featureMeans = new double[featureCount];
        this.featureStdDevs = new double[featureCount];
        
        // Initialize transient fields
        this.lock = new ReentrantReadWriteLock();
        this.pendingSamples = new AtomicInteger(0);
        this.incrementalSamples = new ConcurrentLinkedQueue<>();
    }
    
    /**
     * Initialize or reinitialize transient fields after deserialization
     */
    public void initTransientFields() {
        if (lock == null) {
            new ReentrantReadWriteLock();
        }
        if (pendingSamples == null) {
            new AtomicInteger(0);
        }
        if (incrementalSamples == null) {
            incrementalSamples = new ConcurrentLinkedQueue<>();
        }
    }
    
    /**
     * Add a new training sample
     * 
     * @param features Feature vector
     * @param target Target value
     */
    public void addSample(double[] features, double target) {
        if (features == null || features.length != featureCount) {
            throw new IllegalArgumentException("Invalid feature vector");
        }
        
        if (useIncrementalLearning) {
            // For incremental learning, add to queue
            incrementalSamples.add(new TrainingSample(features.clone(), target));
            pendingSamples.incrementAndGet();
        } else {
            // For batch learning, add directly to storage
            lock.writeLock().lock();
            try {
                // If we've reached max capacity, remove oldest sample
                if (featureVectors.size() >= maxSampleCount) {
                    featureVectors.remove(0);
                    targetValues.remove(0);
                }
                
                // Add new sample
                featureVectors.add(features.clone());
                targetValues.add(target);
            } finally {
                lock.writeLock().unlock();
            }
        }
    }
    
    /**
     * Add a new training sample from metrics
     * 
     * @param metrics Metrics object containing features
     * @param target Target value
     */
    public void addSample(Metrics metrics, double target) {
        if (metrics == null) {
            throw new IllegalArgumentException("Metrics cannot be null");
        }
        
        double[] features = metrics.toFeatureVector();
        addSample(features, target);
    }
    
    /**
     * Get the next batch of incremental samples
     * 
     * @param maxCount Maximum number of samples to retrieve
     * @return Array containing feature vectors and target values
     */
    public Object[] getNextIncrementalBatch(int maxCount) {
        if (!useIncrementalLearning) {
            throw new IllegalStateException("Incremental learning is not enabled");
        }
        
        List<double[]> batchFeatures = new ArrayList<>();
        List<Double> batchTargets = new ArrayList<>();
        int count = 0;
        
        // Retrieve up to maxCount samples from the queue
        while (count < maxCount && !incrementalSamples.isEmpty()) {
            TrainingSample sample = incrementalSamples.poll();
            if (sample != null) {
                batchFeatures.add(sample.features);
                batchTargets.add(sample.target);
                count++;
                pendingSamples.decrementAndGet();
            }
        }
        
        if (batchFeatures.isEmpty()) {
            return null;
        }
        
        // Convert lists to arrays
        double[][] featuresArray = batchFeatures.toArray(new double[0][]);
        double[] targetsArray = new double[batchTargets.size()];
        for (int i = 0; i < batchTargets.size(); i++) {
            targetsArray[i] = batchTargets.get(i);
        }
        
        return new Object[] { featuresArray, targetsArray };
    }
    
    /**
     * Get the number of pending incremental samples
     * 
     * @return Pending sample count
     */
    public int getPendingSampleCount() {
        return pendingSamples.get();
    }
    
    /**
     * Get the current number of stored samples
     * 
     * @return Sample count
     */
    public int getSampleCount() {
        if (useIncrementalLearning) {
            return pendingSamples.get();
        } else {
            lock.readLock().lock();
            try {
                return featureVectors.size();
            } finally {
                lock.readLock().unlock();
            }
        }
    }
    
    /**
     * Create training and validation datasets from collected samples
     * 
     * @param validationRatio Ratio of samples to use for validation (0-1)
     * @return Array containing training and validation datasets
     */
    public Object[] createTrainValidationSplit(double validationRatio) {
        if (validationRatio < 0 || validationRatio >= 1) {
            throw new IllegalArgumentException("Validation ratio must be between 0 and 1");
        }
        
        lock.readLock().lock();
        try {
            int sampleCount = featureVectors.size();
            if (sampleCount == 0) {
                return null;
            }
            
            // Determine split sizes
            int validationSize = Math.max(1, (int)(sampleCount * validationRatio));
            int trainingSize = sampleCount - validationSize;
            
            if (trainingSize <= 0) {
                return null;
            }
            
            // Create randomly shuffled indices
            List<Integer> indices = new ArrayList<>(sampleCount);
            for (int i = 0; i < sampleCount; i++) {
                indices.add(i);
            }
            java.util.Collections.shuffle(indices);
            
            // Create training set
            double[][] trainingFeatures = new double[trainingSize][];
            double[] trainingTargets = new double[trainingSize];
            
            for (int i = 0; i < trainingSize; i++) {
                int idx = indices.get(i);
                trainingFeatures[i] = featureVectors.get(idx);
                trainingTargets[i] = targetValues.get(idx);
            }
            
            // Create validation set
            double[][] validationFeatures = new double[validationSize][];
            double[] validationTargets = new double[validationSize];
            
            for (int i = 0; i < validationSize; i++) {
                int idx = indices.get(trainingSize + i);
                validationFeatures[i] = featureVectors.get(idx);
                validationTargets[i] = targetValues.get(idx);
            }
            
            return new Object[] {
                trainingFeatures, trainingTargets,
                validationFeatures, validationTargets
            };
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Create bootstrap samples for random forest training
     * 
     * @param bootstrapSize Number of samples in each bootstrap
     * @param bootstrapCount Number of bootstrap samples to create
     * @return Array of bootstrap samples
     */
    public Object[][] createBootstrapSamples(int bootstrapSize, int bootstrapCount) {
        lock.readLock().lock();
        try {
            int sampleCount = featureVectors.size();
            if (sampleCount == 0) {
                return null;
            }
            
            // Adjust bootstrap size if needed
            bootstrapSize = Math.min(bootstrapSize, sampleCount);
            
            Object[][] bootstrapSamples = new Object[bootstrapCount][2];
            
            for (int b = 0; b < bootstrapCount; b++) {
                double[][] bootstrapFeatures = new double[bootstrapSize][];
                double[] bootstrapTargets = new double[bootstrapSize];
                
                // Sample with replacement
                for (int i = 0; i < bootstrapSize; i++) {
                    int idx = random.nextInt(sampleCount);
                    bootstrapFeatures[i] = featureVectors.get(idx).clone();
                    bootstrapTargets[i] = targetValues.get(idx);
                }
                
                bootstrapSamples[b][0] = bootstrapFeatures;
                bootstrapSamples[b][1] = bootstrapTargets;
            }
            
            return bootstrapSamples;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Compute feature scaling parameters (mean and standard deviation)
     * from the current samples
     */
    public void computeScalingParameters() {
        lock.readLock().lock();
        try {
            int sampleCount = featureVectors.size();
            if (sampleCount == 0) {
                return;
            }
            
            // Initialize arrays
            double[] sums = new double[featureCount];
            double[] sumSquares = new double[featureCount];
            
            // Compute sums and sum of squares
            for (double[] features : featureVectors) {
                for (int j = 0; j < featureCount; j++) {
                    sums[j] += features[j];
                    sumSquares[j] += features[j] * features[j];
                }
            }
            
            // Compute means and standard deviations
            for (int j = 0; j < featureCount; j++) {
                featureMeans[j] = sums[j] / sampleCount;
                
                double variance = (sumSquares[j] / sampleCount) - (featureMeans[j] * featureMeans[j]);
                featureStdDevs[j] = Math.sqrt(Math.max(1e-8, variance)); // Avoid division by zero
            }
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Apply feature scaling to a feature vector
     * 
     * @param features Raw feature vector
     * @return Scaled feature vector
     */
    public double[] scaleFeatures(double[] features) {
        if (features == null || features.length != featureCount) {
            throw new IllegalArgumentException("Invalid feature vector");
        }
        
        double[] scaled = new double[featureCount];
        
        for (int j = 0; j < featureCount; j++) {
            if (featureStdDevs[j] > 0) {
                scaled[j] = (features[j] - featureMeans[j]) / featureStdDevs[j];
            } else {
                scaled[j] = features[j] - featureMeans[j];
            }
        }
        
        return scaled;
    }
    
    /**
     * Scale all stored feature vectors using the current scaling parameters
     */
    public void scaleAllFeatures() {
        // Ensure scaling parameters are computed
        computeScalingParameters();
        
        lock.writeLock().lock();
        try {
            // Scale each feature vector
            for (int i = 0; i < featureVectors.size(); i++) {
                featureVectors.set(i, scaleFeatures(featureVectors.get(i)));
            }
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Create and return mini-batches from stored samples
     * 
     * @return List of mini-batches
     */
    public List<Object[]> createMiniBatches() {
        lock.readLock().lock();
        try {
            int sampleCount = featureVectors.size();
            if (sampleCount == 0) {
                return new ArrayList<>();
            }
            
            // Create shuffled indices
            List<Integer> indices = new ArrayList<>(sampleCount);
            for (int i = 0; i < sampleCount; i++) {
                indices.add(i);
            }
            java.util.Collections.shuffle(indices);
            
            // Calculate number of batches
            int batchCount = (int) Math.ceil((double) sampleCount / batchSize);
            List<Object[]> batches = new ArrayList<>(batchCount);
            
            for (int b = 0; b < batchCount; b++) {
                int start = b * batchSize;
                int end = Math.min(start + batchSize, sampleCount);
                int currentBatchSize = end - start;
                
                double[][] batchFeatures = new double[currentBatchSize][];
                double[] batchTargets = new double[currentBatchSize];
                
                for (int i = 0; i < currentBatchSize; i++) {
                    int idx = indices.get(start + i);
                    batchFeatures[i] = featureVectors.get(idx);
                    batchTargets[i] = targetValues.get(idx);
                }
                
                batches.add(new Object[] { batchFeatures, batchTargets });
            }
            
            return batches;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Clear all stored samples
     */
    public void clearSamples() {
        lock.writeLock().lock();
        try {
            featureVectors.clear();
            targetValues.clear();
        } finally {
            lock.writeLock().unlock();
        }
        
        if (useIncrementalLearning) {
            incrementalSamples.clear();
            pendingSamples.set(0);
        }
    }
    
    /**
     * Check if a minimum number of samples is available
     * 
     * @param minCount Minimum required sample count
     * @return true if at least minCount samples are available
     */
    public boolean hasSufficientSamples(int minCount) {
        if (useIncrementalLearning) {
            return pendingSamples.get() >= minCount;
        } else {
            lock.readLock().lock();
            try {
                return featureVectors.size() >= minCount;
            } finally {
                lock.readLock().unlock();
            }
        }
    }
    
    /**
     * Get feature means for scaling
     * 
     * @return Array of feature means
     */
    public double[] getFeatureMeans() {
        return featureMeans.clone();
    }
    
    /**
     * Get feature standard deviations for scaling
     * 
     * @return Array of feature standard deviations
     */
    public double[] getFeatureStdDevs() {
        return featureStdDevs.clone();
    }
    
    /**
     * Set feature scaling parameters externally
     * 
     * @param means Array of feature means
     * @param stdDevs Array of feature standard deviations
     */
    public void setScalingParameters(double[] means, double[] stdDevs) {
        if (means == null || stdDevs == null || means.length != featureCount || stdDevs.length != featureCount) {
            throw new IllegalArgumentException("Invalid scaling parameters");
        }
        
        System.arraycopy(means, 0, featureMeans, 0, featureCount);
        System.arraycopy(stdDevs, 0, featureStdDevs, 0, featureCount);
    }
    
    /**
     * Inner class representing a training sample
     */
    private static class TrainingSample implements Serializable {
        private static final long serialVersionUID = 1L;
        
        final double[] features;
        final double target;
        
        TrainingSample(double[] features, double target) {
            this.features = features;
            this.target = target;
        }
    }
}