package com.perphproctor.analyzer;

import com.perphproctor.common.Constants;
import com.perphproctor.common.Metrics;
import com.perphproctor.common.Utils;
import com.perphproctor.common.WorkloadTypes;
import com.perphproctor.common.WorkloadTypes.BatchJobType;
import com.perphproctor.common.WorkloadTypes.WorkloadProfile;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.api.records.Resource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Analyzer is responsible for analyzing workload resource patterns and sensitivity.
 * It runs on a dedicated server and determines resource characteristics of newly
 * submitted jobs through efficient sampling during initial execution phases.
 * 
 * This class implements the functionality described in the Analyzer module of
 * the PerphProctor framework.
 */
public class Analyzer {
    
    private static final Logger LOG = LoggerFactory.getLogger(Analyzer.class);
    
    // Sampling configuration
    private final int samplingIntervalMs;
    private final int minSamplesRequired;
    private final int maxSamplingTimeMs;
    
    // Storage for workload profiles
    private final Map<String, WorkloadProfile> workloadProfiles;
    
    // Active sampling jobs
    private final Map<String, JobSamplingState> activeSamplingJobs;
    
    // Classification models
    private final ResourcePatternDetector resourcePatternDetector;
    private final SensitivityClassifier sensitivityClassifier;
    
    // Analysis executor
    private final ExecutorService analysisExecutor;
    
    // Thread safety
    private final ReentrantReadWriteLock lock;
    
    // Analysis state
    private final AtomicBoolean isRunning;
    
    /**
     * Constructor with default configuration
     */
    public Analyzer() {
        this(
            5000,  // 5 second sampling interval
            12,    // 12 minimum samples
            300000 // 5 minute max sampling time
        );
    }
    
    /**
     * Constructor with custom configuration
     * 
     * @param samplingIntervalMs Interval between samples in milliseconds
     * @param minSamplesRequired Minimum number of samples to collect
     * @param maxSamplingTimeMs Maximum sampling time in milliseconds
     */
    public Analyzer(int samplingIntervalMs, int minSamplesRequired, int maxSamplingTimeMs) {
        this.samplingIntervalMs = samplingIntervalMs;
        this.minSamplesRequired = minSamplesRequired;
        this.maxSamplingTimeMs = maxSamplingTimeMs;
        
        this.workloadProfiles = new ConcurrentHashMap<>();
        this.activeSamplingJobs = new ConcurrentHashMap<>();
        
        this.resourcePatternDetector = new ResourcePatternDetector();
        this.sensitivityClassifier = new SensitivityClassifier();
        
        this.analysisExecutor = Executors.newFixedThreadPool(2);
        this.lock = new ReentrantReadWriteLock();
        this.isRunning = new AtomicBoolean(false);
        
        LOG.info("Analyzer initialized with samplingInterval={}, minSamples={}, maxSamplingTime={}",
                samplingIntervalMs, minSamplesRequired, maxSamplingTimeMs);
    }
    
    /**
     * Start the analyzer
     */
    public void start() {
        if (isRunning.compareAndSet(false, true)) {
            LOG.info("Starting Analyzer");
        } else {
            LOG.warn("Analyzer already running");
        }
    }
    
    /**
     * Stop the analyzer
     */
    public void stop() {
        if (isRunning.compareAndSet(true, false)) {
            LOG.info("Stopping Analyzer");
            
            analysisExecutor.shutdown();
            try {
                if (!analysisExecutor.awaitTermination(30, TimeUnit.SECONDS)) {
                    analysisExecutor.shutdownNow();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                analysisExecutor.shutdownNow();
            }
        } else {
            LOG.warn("Analyzer already stopped");
        }
    }
    
    /**
     * Start analyzing a new job
     * 
     * @param jobId Job identifier
     * @param jobName Job name
     * @return true if analysis started successfully
     */
    public boolean startJobAnalysis(String jobId, String jobName) {
        if (!isRunning.get()) {
            LOG.warn("Cannot start job analysis: Analyzer not running");
            return false;
        }
        
        if (jobId == null || jobName == null) {
            LOG.warn("Cannot start job analysis: Invalid parameters");
            return false;
        }
        
        // Check if job is already profiled
        if (workloadProfiles.containsKey(jobId)) {
            LOG.info("Job {} already profiled, skipping analysis", jobId);
            return true;
        }
        
        // Check if job is already being sampled
        if (activeSamplingJobs.containsKey(jobId)) {
            LOG.info("Job {} already being sampled", jobId);
            return true;
        }
        
        // Create new sampling state
        JobSamplingState samplingState = new JobSamplingState(jobId, jobName);
        
        // Add to active sampling jobs
        lock.writeLock().lock();
        try {
            activeSamplingJobs.put(jobId, samplingState);
        } finally {
            lock.writeLock().unlock();
        }
        
        // Start sampling task
        analysisExecutor.submit(() -> sampleJobResources(jobId));
        
        LOG.info("Started analysis for job {}: {}", jobId, jobName);
        return true;
    }
    
    /**
     * Sample job resources over time
     * 
     * @param jobId Job identifier
     */
    private void sampleJobResources(String jobId) {
        JobSamplingState samplingState = activeSamplingJobs.get(jobId);
        if (samplingState == null) {
            LOG.warn("Cannot sample job {}: Job not found in active sampling jobs", jobId);
            return;
        }
        
        LOG.info("Starting resource sampling for job {}", jobId);
        
        long startTime = System.currentTimeMillis();
        long endTime = startTime + maxSamplingTimeMs;
        
        while (System.currentTimeMillis() < endTime) {
            // Check if analyzer is still running
            if (!isRunning.get()) {
                LOG.info("Stopping sampling for job {} because analyzer is shutting down", jobId);
                break;
            }
            
            // Collect sample
            Metrics metrics = collectJobMetrics(jobId);
            if (metrics != null) {
                samplingState.addSample(metrics);
                LOG.debug("Collected sample {} for job {}", 
                         samplingState.getSampleCount(), jobId);
            }
            
            // Check if we have enough samples
            if (samplingState.getSampleCount() >= minSamplesRequired) {
                LOG.info("Collected sufficient samples ({}) for job {}", 
                        samplingState.getSampleCount(), jobId);
                break;
            }
            
            // Wait for next sampling interval
            try {
                Thread.sleep(samplingIntervalMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                LOG.warn("Sampling for job {} interrupted", jobId);
                break;
            }
        }
        
        // Process collected samples
        if (samplingState.getSampleCount() >= minSamplesRequired) {
            processJobSamples(samplingState);
        } else {
            LOG.warn("Insufficient samples for job {}: {} collected, {} required",
                    jobId, samplingState.getSampleCount(), minSamplesRequired);
        }
        
        // Remove from active sampling jobs
        lock.writeLock().lock();
        try {
            activeSamplingJobs.remove(jobId);
        } finally {
            lock.writeLock().unlock();
        }
        
        LOG.info("Completed sampling for job {}", jobId);
    }
    
    /**
     * Collect metrics for a job
     * 
     * @param jobId Job identifier
     * @return Collected metrics or null if collection failed
     */
    private Metrics collectJobMetrics(String jobId) {
        // In a real implementation, this would collect actual metrics from the running job
        // For this implementation, we use simulated metrics
        
        try {
            Metrics metrics = new Metrics();
            
            // Simulate resource metrics collection
            metrics.setCpuUtilization(30 + Math.random() * 50); // 30-80% CPU
            metrics.setMemoryUtilization(40 + Math.random() * 30); // 40-70% Memory
            metrics.setLlcUtilization(20 + Math.random() * 40); // 20-60% LLC
            metrics.setMemoryBandwidth(Math.random() * 10000); // 0-10000 MB/s
            metrics.setIoThroughput(Math.random() * 500); // 0-500 IOPS
            
            return metrics;
        } catch (Exception e) {
            LOG.error("Error collecting metrics for job {}: {}", jobId, e.getMessage());
            return null;
        }
    }
    
    /**
     * Process collected job samples to determine resource pattern and sensitivity
     * 
     * @param samplingState Job sampling state with collected samples
     */
    private void processJobSamples(JobSamplingState samplingState) {
        String jobId = samplingState.getJobId();
        String jobName = samplingState.getJobName();
        List<Metrics> samples = samplingState.getSamples();
        
        LOG.info("Processing {} samples for job {}", samples.size(), jobId);
        
        try {
            // Detect resource utilization pattern
            Map<String, Double> resourceUtilization = calculateResourceUtilization(samples);
            
            // Classify batch type
            BatchJobType batchType = resourcePatternDetector.detectBatchType(resourceUtilization);
            
            // Create workload profile
            WorkloadProfile profile = WorkloadTypes.createBatchProfile(jobId, jobName, batchType);
            
            // Analyze resource sensitivity
            Map<String, Integer> sensitivity = sensitivityClassifier.classifySensitivity(
                    resourceUtilization, batchType);
            
            // Update profile with sensitivity information
            profile.setCpuSensitivity(sensitivity.getOrDefault("cpu", 5));
            profile.setMemorySensitivity(sensitivity.getOrDefault("memory", 5));
            profile.setLlcSensitivity(sensitivity.getOrDefault("llc", 5));
            profile.setMbwSensitivity(sensitivity.getOrDefault("mbw", 5));
            profile.setIoSensitivity(sensitivity.getOrDefault("io", 5));
            
            // Store profile
            lock.writeLock().lock();
            try {
                workloadProfiles.put(jobId, profile);
            } finally {
                lock.writeLock().unlock();
            }
            
            LOG.info("Job {} classified as {} with sensitivity cpu={}, memory={}, llc={}, mbw={}, io={}",
                    jobId, batchType, 
                    sensitivity.getOrDefault("cpu", 5),
                    sensitivity.getOrDefault("memory", 5),
                    sensitivity.getOrDefault("llc", 5),
                    sensitivity.getOrDefault("mbw", 5),
                    sensitivity.getOrDefault("io", 5));
        } catch (Exception e) {
            LOG.error("Error processing samples for job {}: {}", jobId, e.getMessage());
        }
    }
    
    /**
     * Calculate average resource utilization from samples
     * 
     * @param samples List of metrics samples
     * @return Map of resource names to utilization values
     */
    private Map<String, Double> calculateResourceUtilization(List<Metrics> samples) {
        Map<String, Double> utilization = new HashMap<>();
        
        if (samples == null || samples.isEmpty()) {
            return utilization;
        }
        
        double totalCpu = 0;
        double totalMemory = 0;
        double totalLlc = 0;
        double totalMbw = 0;
        double totalIo = 0;
        
        for (Metrics metrics : samples) {
            totalCpu += metrics.getCpuUtilization();
            totalMemory += metrics.getMemoryUtilization();
            totalLlc += metrics.getLlcUtilization();
            totalMbw += metrics.getMemoryBandwidth();
            totalIo += metrics.getIoThroughput();
        }
        
        int count = samples.size();
        
        utilization.put("cpu", totalCpu / count);
        utilization.put("memory", totalMemory / count);
        utilization.put("llc", totalLlc / count);
        utilization.put("mbw", totalMbw / count);
        utilization.put("io", totalIo / count);
        
        return utilization;
    }
    
    /**
     * Get the workload profile for a job
     * 
     * @param jobId Job identifier
     * @return Workload profile or null if not found
     */
    public WorkloadProfile getWorkloadProfile(String jobId) {
        if (jobId == null) {
            return null;
        }
        
        lock.readLock().lock();
        try {
            return workloadProfiles.get(jobId);
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Get the batch job type for a job
     * 
     * @param jobId Job identifier
     * @return Batch job type or UNKNOWN if not found
     */
    public BatchJobType getBatchJobType(String jobId) {
        WorkloadProfile profile = getWorkloadProfile(jobId);
        return profile != null ? profile.getBatchType() : BatchJobType.UNKNOWN;
    }
    
    /**
     * Get the resource sensitivity for a job
     * 
     * @param jobId Job identifier
     * @return Map of resource names to sensitivity values (0-10)
     */
    public Map<String, Integer> getResourceSensitivity(String jobId) {
        WorkloadProfile profile = getWorkloadProfile(jobId);
        
        if (profile == null) {
            return new HashMap<>();
        }
        
        Map<String, Integer> sensitivity = new HashMap<>();
        sensitivity.put("cpu", profile.getCpuSensitivity());
        sensitivity.put("memory", profile.getMemorySensitivity());
        sensitivity.put("llc", profile.getLlcSensitivity());
        sensitivity.put("mbw", profile.getMbwSensitivity());
        sensitivity.put("io", profile.getIoSensitivity());
        
        return sensitivity;
    }
    
    /**
     * Check if analysis is in progress for a job
     * 
     * @param jobId Job identifier
     * @return true if job is being analyzed
     */
    public boolean isJobBeingAnalyzed(String jobId) {
        if (jobId == null) {
            return false;
        }
        
        return activeSamplingJobs.containsKey(jobId);
    }
    
    /**
     * Get the number of jobs currently being analyzed
     * 
     * @return Count of active sampling jobs
     */
    public int getActiveJobCount() {
        return activeSamplingJobs.size();
    }
    
    /**
     * Get the number of profiled jobs
     * 
     * @return Count of profiled jobs
     */
    public int getProfiledJobCount() {
        return workloadProfiles.size();
    }
    
    /**
     * Inner class for resource pattern detection
     */
    private static class ResourcePatternDetector {
        
        /**
         * Detect batch job type based on resource utilization pattern
         * 
         * @param resourceUtilization Map of resource names to utilization values
         * @return Detected batch job type
         */
        public BatchJobType detectBatchType(Map<String, Double> resourceUtilization) {
            if (resourceUtilization == null || resourceUtilization.isEmpty()) {
                return BatchJobType.UNKNOWN;
            }
            
            double cpuUtil = resourceUtilization.getOrDefault("cpu", 0.0);
            double memoryUtil = resourceUtilization.getOrDefault("memory", 0.0);
            double ioUtil = resourceUtilization.getOrDefault("io", 0.0);
            
            // CPU-intensive if CPU utilization is highest
            if (cpuUtil > 70 && ioUtil < 40) {
                return BatchJobType.CPU_INTENSIVE;
            }
            
            // IO-intensive if IO throughput is highest
            if (ioUtil > 70 && cpuUtil < 40) {
                return BatchJobType.IO_INTENSIVE;
            }
            
            // Memory-intensive if memory utilization is highest
            if (memoryUtil > 70 && cpuUtil < 60 && ioUtil < 60) {
                return BatchJobType.MEMORY_INTENSIVE;
            }
            
            // Hybrid if both CPU and IO are significant
            if (cpuUtil > 50 && ioUtil > 50) {
                return BatchJobType.HYBRID;
            }
            
            // Default to unknown
            return BatchJobType.UNKNOWN;
        }
    }
    
    /**
     * Inner class for sensitivity classification
     */
    private static class SensitivityClassifier {
        
        /**
         * Classify resource sensitivity based on utilization pattern
         * 
         * @param resourceUtilization Map of resource names to utilization values
         * @param batchType Detected batch job type
         * @return Map of resource names to sensitivity values (0-10)
         */
        public Map<String, Integer> classifySensitivity(
                Map<String, Double> resourceUtilization, BatchJobType batchType) {
            
            Map<String, Integer> sensitivity = new HashMap<>();
            
            // Default sensitivity values
            sensitivity.put("cpu", 5);
            sensitivity.put("memory", 5);
            sensitivity.put("llc", 5);
            sensitivity.put("mbw", 5);
            sensitivity.put("io", 5);
            
            if (resourceUtilization == null || resourceUtilization.isEmpty()) {
                return sensitivity;
            }
            
            // Adjust sensitivity based on batch type
            switch (batchType) {
                case CPU_INTENSIVE:
                    sensitivity.put("cpu", 9);
                    sensitivity.put("llc", 7);
                    sensitivity.put("io", 2);
                    break;
                    
                case IO_INTENSIVE:
                    sensitivity.put("io", 9);
                    sensitivity.put("cpu", 3);
                    sensitivity.put("llc", 3);
                    break;
                    
                case MEMORY_INTENSIVE:
                    sensitivity.put("memory", 9);
                    sensitivity.put("mbw", 8);
                    sensitivity.put("llc", 6);
                    break;
                    
                case HYBRID:
                    sensitivity.put("cpu", 7);
                    sensitivity.put("io", 7);
                    sensitivity.put("memory", 6);
                    break;
                    
                default:
                    // Further fine-tune based on actual utilization values
                    double cpuUtil = resourceUtilization.getOrDefault("cpu", 0.0);
                    double memoryUtil = resourceUtilization.getOrDefault("memory", 0.0);
                    double ioUtil = resourceUtilization.getOrDefault("io", 0.0);
                    
                    sensitivity.put("cpu", convertUtilizationToSensitivity(cpuUtil));
                    sensitivity.put("memory", convertUtilizationToSensitivity(memoryUtil));
                    sensitivity.put("io", convertUtilizationToSensitivity(ioUtil));
            }
            
            return sensitivity;
        }
        
        /**
         * Convert utilization percentage to sensitivity scale (0-10)
         * 
         * @param utilization Utilization percentage (0-100)
         * @return Sensitivity value (0-10)
         */
        private int convertUtilizationToSensitivity(double utilization) {
            // Simple linear mapping from 0-100 to 0-10
            int sensitivity = (int) Math.round(utilization / 10.0);
            
            // Ensure within bounds
            return Math.max(0, Math.min(10, sensitivity));
        }
    }
    
    /**
     * Inner class for job sampling state
     */
    private static class JobSamplingState {
        private final String jobId;
        private final String jobName;
        private final List<Metrics> samples;
        private final long startTime;
        
        /**
         * Constructor
         * 
         * @param jobId Job identifier
         * @param jobName Job name
         */
        public JobSamplingState(String jobId, String jobName) {
            this.jobId = jobId;
            this.jobName = jobName;
            this.samples = new ArrayList<>();
            this.startTime = System.currentTimeMillis();
        }
        
        /**
         * Add a metrics sample
         * 
         * @param metrics Metrics sample
         */
        public void addSample(Metrics metrics) {
            if (metrics != null) {
                samples.add(metrics);
            }
        }
        
        /**
         * Get the job identifier
         * 
         * @return Job identifier
         */
        public String getJobId() {
            return jobId;
        }
        
        /**
         * Get the job name
         * 
         * @return Job name
         */
        public String getJobName() {
            return jobName;
        }
        
        /**
         * Get collected samples
         * 
         * @return List of metrics samples
         */
        public List<Metrics> getSamples() {
            return new ArrayList<>(samples);
        }
        
        /**
         * Get the number of collected samples
         * 
         * @return Sample count
         */
        public int getSampleCount() {
            return samples.size();
        }
        
        /**
         * Get the sampling start time
         * 
         * @return Start time in milliseconds
         */
        public long getStartTime() {
            return startTime;
        }
        
        /**
         * Get the elapsed sampling time
         * 
         * @return Elapsed time in milliseconds
         */
        public long getElapsedTime() {
            return System.currentTimeMillis() - startTime;
        }
    }
}