package com.perphproctor.analyzer;

import com.perphproctor.common.Metrics;
import com.perphproctor.common.Utils;
import com.perphproctor.common.WorkloadTypes;
import com.perphproctor.common.WorkloadTypes.BatchJobType;
import com.perphproctor.common.WorkloadTypes.WorkloadProfile;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * JobProfiler is responsible for creating, managing, and storing workload 
 * profiles based on resource usage observations. It captures the essential
 * characteristics of jobs to enable intelligent scheduling decisions.
 * 
 * This class is part of the Analyzer module in the PerphProctor framework
 * and works with ResourcePatternDetector to classify workloads.
 */
public class JobProfiler implements Serializable {
    
    private static final long serialVersionUID = 1L;
    private static final Logger LOG = LoggerFactory.getLogger(JobProfiler.class);
    
    // Default profile storage location
    private static final String DEFAULT_PROFILE_DIR = "/tmp/perphproctor/profiles";
    
    // Minimum samples needed for profiling
    private static final int MIN_SAMPLES_FOR_PROFILE = 10;
    
    // Profile storage
    private final Map<String, WorkloadProfile> profiles;
    
    // Resource pattern detector
    private final ResourcePatternDetector patternDetector;
    
    // Sensitivity classifier
    private final SensitivityClassifier sensitivityClassifier;
    
    // Active profiling jobs
    private final Map<String, ProfilingJob> activeJobs;
    
    // Profiling statistics
    private final AtomicInteger totalProfiledJobs;
    private final AtomicInteger failedProfilingJobs;
    
    // Profile storage directory
    private final String profileStorageDir;
    
    /**
     * Constructor with default profile storage location
     */
    public JobProfiler() {
        this(DEFAULT_PROFILE_DIR);
    }
    
    /**
     * Constructor with custom profile storage location
     * 
     * @param profileStorageDir Directory to store profiles
     */
    public JobProfiler(String profileStorageDir) {
        this.profiles = new ConcurrentHashMap<>();
        this.patternDetector = new ResourcePatternDetector();
        this.sensitivityClassifier = new SensitivityClassifier();
        this.activeJobs = new ConcurrentHashMap<>();
        this.totalProfiledJobs = new AtomicInteger(0);
        this.failedProfilingJobs = new AtomicInteger(0);
        this.profileStorageDir = profileStorageDir;
        
        // Ensure profile directory exists
        createProfileDirectory();
        
        // Load existing profiles
        loadProfiles();
        
        LOG.info("JobProfiler initialized with profile storage at: {}", profileStorageDir);
        LOG.info("Loaded {} existing profiles", profiles.size());
    }
    
    /**
     * Start profiling a job
     * 
     * @param jobId Job identifier
     * @param jobName Job name
     * @return true if profiling started successfully
     */
    public boolean startProfiling(String jobId, String jobName) {
        if (jobId == null || jobName == null) {
            LOG.warn("Cannot start profiling: Invalid parameters");
            return false;
        }
        
        // Check if already profiled
        if (profiles.containsKey(jobId)) {
            LOG.info("Job {} already profiled", jobId);
            return true;
        }
        
        // Check if already profiling
        if (activeJobs.containsKey(jobId)) {
            LOG.info("Job {} already being profiled", jobId);
            return true;
        }
        
        // Create new profiling job
        ProfilingJob profilingJob = new ProfilingJob(jobId, jobName);
        activeJobs.put(jobId, profilingJob);
        
        LOG.info("Started profiling job {}: {}", jobId, jobName);
        return true;
    }
    
    /**
     * Add a metrics sample for a job being profiled
     * 
     * @param jobId Job identifier
     * @param metrics Metrics sample
     */
    public void addMetricsSample(String jobId, Metrics metrics) {
        if (jobId == null || metrics == null) {
            return;
        }
        
        ProfilingJob job = activeJobs.get(jobId);
        if (job == null) {
            LOG.debug("No active profiling job found for {}", jobId);
            return;
        }
        
        // Add metrics sample
        job.addSample(metrics);
        LOG.debug("Added metrics sample for job {}: {}", jobId, metrics);
        
        // Check if we have enough samples to complete profiling
        if (job.getSampleCount() >= MIN_SAMPLES_FOR_PROFILE) {
            completeJobProfiling(jobId);
        }
    }
    
    /**
     * Complete profiling for a job
     * 
     * @param jobId Job identifier
     * @return Created profile or null if profiling failed
     */
    public WorkloadProfile completeJobProfiling(String jobId) {
        if (jobId == null) {
            return null;
        }
        
        ProfilingJob job = activeJobs.remove(jobId);
        if (job == null) {
            LOG.warn("No active profiling job found for {}", jobId);
            return null;
        }
        
        try {
            LOG.info("Completing profile for job {} with {} samples", 
                    jobId, job.getSampleCount());
            
            // Get batch job type
            BatchJobType batchType = patternDetector.detectBatchType(job.getSamples());
            
            // Create profile
            WorkloadProfile profile = WorkloadTypes.createBatchProfile(
                    jobId, job.getJobName(), batchType);
            
            // Analyze resource sensitivity
            Map<String, Double> avgUtilization = calculateAverageUtilization(job.getSamples());
            Map<String, Integer> sensitivity = sensitivityClassifier.classifySensitivity(
                    avgUtilization, batchType);
            
            // Update profile with sensitivity information
            profile.setCpuSensitivity(sensitivity.getOrDefault("cpu", 5));
            profile.setMemorySensitivity(sensitivity.getOrDefault("memory", 5));
            profile.setLlcSensitivity(sensitivity.getOrDefault("llc", 5));
            profile.setMbwSensitivity(sensitivity.getOrDefault("mbw", 5));
            profile.setIoSensitivity(sensitivity.getOrDefault("io", 5));
            
            // Set mean utilization
            profile.setMeanCpuUtil(avgUtilization.getOrDefault("cpu", 0.0));
            profile.setMeanMemoryUtil(avgUtilization.getOrDefault("memory", 0.0));
            profile.setMeanLlcUtil(avgUtilization.getOrDefault("llc", 0.0));
            profile.setMeanMbwUtil(avgUtilization.getOrDefault("mbw", 0.0));
            profile.setMeanIoUtil(avgUtilization.getOrDefault("io", 0.0));
            
            // Store profile
            profiles.put(jobId, profile);
            
            // Save profile to disk
            saveProfile(jobId, profile);
            
            // Update statistics
            totalProfiledJobs.incrementAndGet();
            
            LOG.info("Completed profile for job {}: BatchType={}, CPU sensitivity={}, IO sensitivity={}",
                    jobId, batchType, 
                    profile.getCpuSensitivity(), 
                    profile.getIoSensitivity());
            
            return profile;
        } catch (Exception e) {
            LOG.error("Error completing profile for job {}: {}", jobId, e.getMessage());
            failedProfilingJobs.incrementAndGet();
            return null;
        }
    }
    
    /**
     * Calculate average utilization for multiple metrics samples
     * 
     * @param samples List of metrics samples
     * @return Map of resource names to average utilization values
     */
    private Map<String, Double> calculateAverageUtilization(List<Metrics> samples) {
        Map<String, Double> avgUtilization = new HashMap<>();
        
        if (samples == null || samples.isEmpty()) {
            return avgUtilization;
        }
        
        double totalCpu = 0.0;
        double totalMemory = 0.0;
        double totalLlc = 0.0;
        double totalMbw = 0.0;
        double totalIo = 0.0;
        
        for (Metrics metrics : samples) {
            totalCpu += metrics.getCpuUtilization();
            totalMemory += metrics.getMemoryUtilization();
            totalLlc += metrics.getLlcUtilization();
            totalMbw += metrics.getMemoryBandwidth();
            totalIo += metrics.getIoThroughput();
        }
        
        int count = samples.size();
        
        avgUtilization.put("cpu", totalCpu / count);
        avgUtilization.put("memory", totalMemory / count);
        avgUtilization.put("llc", totalLlc / count);
        avgUtilization.put("mbw", totalMbw / count);
        avgUtilization.put("io", totalIo / count);
        
        return avgUtilization;
    }
    
    /**
     * Get the workload profile for a job
     * 
     * @param jobId Job identifier
     * @return Workload profile or null if not found
     */
    public WorkloadProfile getProfile(String jobId) {
        if (jobId == null) {
            return null;
        }
        
        return profiles.get(jobId);
    }
    
    /**
     * Get workload profiles for multiple jobs
     * 
     * @param jobIds List of job identifiers
     * @return Map of job IDs to profiles
     */
    public Map<String, WorkloadProfile> getProfiles(List<String> jobIds) {
        Map<String, WorkloadProfile> result = new HashMap<>();
        
        if (jobIds == null) {
            return result;
        }
        
        for (String jobId : jobIds) {
            WorkloadProfile profile = getProfile(jobId);
            if (profile != null) {
                result.put(jobId, profile);
            }
        }
        
        return result;
    }
    
    /**
     * Check if a job is being profiled
     * 
     * @param jobId Job identifier
     * @return true if job is being profiled
     */
    public boolean isJobBeingProfiled(String jobId) {
        if (jobId == null) {
            return false;
        }
        
        return activeJobs.containsKey(jobId);
    }
    
    /**
     * Check if a job has already been profiled
     * 
     * @param jobId Job identifier
     * @return true if job has been profiled
     */
    public boolean isJobProfiled(String jobId) {
        if (jobId == null) {
            return false;
        }
        
        return profiles.containsKey(jobId);
    }
    
    /**
     * Get all profiled jobs
     * 
     * @return Map of job IDs to profiles
     */
    public Map<String, WorkloadProfile> getAllProfiles() {
        return new HashMap<>(profiles);
    }
    
    /**
     * Delete a profile
     * 
     * @param jobId Job identifier
     * @return true if profile was deleted
     */
    public boolean deleteProfile(String jobId) {
        if (jobId == null) {
            return false;
        }
        
        WorkloadProfile removed = profiles.remove(jobId);
        if (removed != null) {
            // Delete profile file
            File profileFile = getProfileFile(jobId);
            if (profileFile.exists()) {
                boolean deleted = profileFile.delete();
                if (!deleted) {
                    LOG.warn("Failed to delete profile file for job {}", jobId);
                }
            }
            
            LOG.info("Deleted profile for job {}", jobId);
            return true;
        }
        
        return false;
    }
    
    /**
     * Cancel profiling for a job
     * 
     * @param jobId Job identifier
     * @return true if profiling was canceled
     */
    public boolean cancelProfiling(String jobId) {
        if (jobId == null) {
            return false;
        }
        
        ProfilingJob removed = activeJobs.remove(jobId);
        if (removed != null) {
            LOG.info("Canceled profiling for job {}", jobId);
            return true;
        }
        
        return false;
    }
    
    /**
     * Find similar profiles to a given job
     * 
     * @param jobId Job identifier
     * @param maxResults Maximum number of results to return
     * @return List of similar job IDs
     */
    public List<String> findSimilarProfiles(String jobId, int maxResults) {
        if (jobId == null || !profiles.containsKey(jobId)) {
            return new ArrayList<>();
        }
        
        WorkloadProfile targetProfile = profiles.get(jobId);
        List<ProfileSimilarity> similarities = new ArrayList<>();
        
        // Calculate similarity with all other profiles
        for (Map.Entry<String, WorkloadProfile> entry : profiles.entrySet()) {
            String otherJobId = entry.getKey();
            
            // Skip self comparison
            if (otherJobId.equals(jobId)) {
                continue;
            }
            
            WorkloadProfile otherProfile = entry.getValue();
            double similarity = calculateProfileSimilarity(targetProfile, otherProfile);
            
            similarities.add(new ProfileSimilarity(otherJobId, similarity));
        }
        
        // Sort by similarity (highest first)
        similarities.sort((a, b) -> Double.compare(b.similarity, a.similarity));
        
        // Return top N results
        List<String> result = new ArrayList<>();
        int count = Math.min(maxResults, similarities.size());
        
        for (int i = 0; i < count; i++) {
            result.add(similarities.get(i).jobId);
        }
        
        return result;
    }
    
    /**
     * Calculate similarity between two workload profiles
     * 
     * @param profile1 First profile
     * @param profile2 Second profile
     * @return Similarity score (0-1)
     */
    private double calculateProfileSimilarity(WorkloadProfile profile1, WorkloadProfile profile2) {
        // Check if batch types match
        if (profile1.getBatchType() != profile2.getBatchType()) {
            return 0.2; // Low baseline similarity for different types
        }
        
        // Calculate similarity based on resource sensitivity
        double cpuSimilarity = 1.0 - Math.abs(profile1.getCpuSensitivity() - profile2.getCpuSensitivity()) / 10.0;
        double memorySimilarity = 1.0 - Math.abs(profile1.getMemorySensitivity() - profile2.getMemorySensitivity()) / 10.0;
        double llcSimilarity = 1.0 - Math.abs(profile1.getLlcSensitivity() - profile2.getLlcSensitivity()) / 10.0;
        double mbwSimilarity = 1.0 - Math.abs(profile1.getMbwSensitivity() - profile2.getMbwSensitivity()) / 10.0;
        double ioSimilarity = 1.0 - Math.abs(profile1.getIoSensitivity() - profile2.getIoSensitivity()) / 10.0;
        
        // Calculate overall similarity (weighted average)
        return 0.3 * cpuSimilarity + 
               0.2 * memorySimilarity + 
               0.2 * ioSimilarity + 
               0.15 * llcSimilarity + 
               0.15 * mbwSimilarity;
    }
    
    /**
     * Get profile statistics
     * 
     * @return Map of statistics
     */
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new HashMap<>();
        
        stats.put("totalProfiles", profiles.size());
        stats.put("activeProfilingJobs", activeJobs.size());
        stats.put("totalProfiledJobs", totalProfiledJobs.get());
        stats.put("failedProfilingJobs", failedProfilingJobs.get());
        
        // Count profiles by type
        Map<BatchJobType, Integer> typeCount = new HashMap<>();
        for (WorkloadProfile profile : profiles.values()) {
            BatchJobType type = profile.getBatchType();
            typeCount.put(type, typeCount.getOrDefault(type, 0) + 1);
        }
        
        stats.put("profilesByType", typeCount);
        
        return stats;
    }
    
    /**
     * Create profile storage directory
     */
    private void createProfileDirectory() {
        File dir = new File(profileStorageDir);
        if (!dir.exists()) {
            boolean created = dir.mkdirs();
            if (!created) {
                LOG.error("Failed to create profile directory: {}", profileStorageDir);
            }
        }
    }
    
    /**
     * Get profile file for a job
     * 
     * @param jobId Job identifier
     * @return Profile file
     */
    private File getProfileFile(String jobId) {
        return new File(profileStorageDir, "profile_" + jobId + ".dat");
    }
    
    /**
     * Save profile to disk
     * 
     * @param jobId Job identifier
     * @param profile Workload profile
     */
    private void saveProfile(String jobId, WorkloadProfile profile) {
        File profileFile = getProfileFile(jobId);
        
        try (FileOutputStream fos = new FileOutputStream(profileFile);
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            oos.writeObject(profile);
            LOG.debug("Saved profile for job {} to {}", jobId, profileFile.getAbsolutePath());
        } catch (Exception e) {
            LOG.error("Failed to save profile for job {}: {}", jobId, e.getMessage());
        }
    }
    
    /**
     * Load profiles from disk
     */
    private void loadProfiles() {
        File dir = new File(profileStorageDir);
        if (!dir.exists() || !dir.isDirectory()) {
            LOG.warn("Profile directory does not exist: {}", profileStorageDir);
            return;
        }
        
        File[] files = dir.listFiles((d, name) -> name.startsWith("profile_") && name.endsWith(".dat"));
        if (files == null || files.length == 0) {
            LOG.info("No profile files found in {}", profileStorageDir);
            return;
        }
        
        for (File file : files) {
            try (FileInputStream fis = new FileInputStream(file);
                 ObjectInputStream ois = new ObjectInputStream(fis)) {
                
                WorkloadProfile profile = (WorkloadProfile) ois.readObject();
                String jobId = profile.getWorkloadId();
                
                profiles.put(jobId, profile);
                LOG.debug("Loaded profile for job {} from {}", jobId, file.getAbsolutePath());
            } catch (Exception e) {
                LOG.error("Failed to load profile from {}: {}", file.getAbsolutePath(), e.getMessage());
            }
        }
    }
    
    /**
     * Inner class to hold job sampling state
     */
    private static class ProfilingJob implements Serializable {
        private static final long serialVersionUID = 1L;
        
        private final String jobId;
        private final String jobName;
        private final List<Metrics> samples;
        private final long startTime;
        
        public ProfilingJob(String jobId, String jobName) {
            this.jobId = jobId;
            this.jobName = jobName;
            this.samples = new ArrayList<>();
            this.startTime = System.currentTimeMillis();
        }
        
        public void addSample(Metrics metrics) {
            if (metrics != null) {
                samples.add(metrics);
            }
        }
        
        public String getJobId() {
            return jobId;
        }
        
        public String getJobName() {
            return jobName;
        }
        
        public List<Metrics> getSamples() {
            return new ArrayList<>(samples);
        }
        
        public int getSampleCount() {
            return samples.size();
        }
        
        public long getStartTime() {
            return startTime;
        }
        
        public long getElapsedTime() {
            return System.currentTimeMillis() - startTime;
        }
    }
    
    /**
     * Inner class for sensitivity classification
     */
    private static class SensitivityClassifier implements Serializable {
        private static final long serialVersionUID = 1L;
        
        /**
         * Classify resource sensitivity based on utilization pattern and batch type
         * 
         * @param resourceUtilization Map of resource names to utilization values
         * @param batchType Batch job type
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
                    sensitivity.put("memory", 5);
                    sensitivity.put("mbw", 4);
                    sensitivity.put("io", 2);
                    break;
                    
                case IO_INTENSIVE:
                    sensitivity.put("io", 9);
                    sensitivity.put("mbw", 5);
                    sensitivity.put("cpu", 3);
                    sensitivity.put("memory", 4);
                    sensitivity.put("llc", 3);
                    break;
                    
                case MEMORY_INTENSIVE:
                    sensitivity.put("memory", 9);
                    sensitivity.put("mbw", 8);
                    sensitivity.put("llc", 6);
                    sensitivity.put("cpu", 4);
                    sensitivity.put("io", 3);
                    break;
                    
                case HYBRID:
                    sensitivity.put("cpu", 7);
                    sensitivity.put("io", 7);
                    sensitivity.put("memory", 6);
                    sensitivity.put("llc", 5);
                    sensitivity.put("mbw", 6);
                    break;
                    
                default:
                    // Further adjust based on actual utilization values
                    adjustSensitivityFromUtilization(sensitivity, resourceUtilization);
            }
            
            return sensitivity;
        }
        
        /**
         * Adjust sensitivity values based on resource utilization
         * 
         * @param sensitivity Map of resource names to sensitivity values
         * @param utilization Map of resource names to utilization values
         */
        private void adjustSensitivityFromUtilization(
                Map<String, Integer> sensitivity, Map<String, Double> utilization) {
            
            // CPU sensitivity
            double cpuUtil = utilization.getOrDefault("cpu", 0.0);
            sensitivity.put("cpu", convertUtilizationToSensitivity(cpuUtil));
            
            // Memory sensitivity
            double memoryUtil = utilization.getOrDefault("memory", 0.0);
            sensitivity.put("memory", convertUtilizationToSensitivity(memoryUtil));
            
            // LLC sensitivity
            double llcUtil = utilization.getOrDefault("llc", 0.0);
            sensitivity.put("llc", convertUtilizationToSensitivity(llcUtil));
            
            // MBW sensitivity
            double mbwUtil = utilization.getOrDefault("mbw", 0.0) / 100.0;
            sensitivity.put("mbw", convertUtilizationToSensitivity(mbwUtil));
            
            // IO sensitivity
            double ioUtil = utilization.getOrDefault("io", 0.0) / 50.0;
            sensitivity.put("io", convertUtilizationToSensitivity(ioUtil));
        }
        
        /**
         * Convert utilization percentage to sensitivity scale (0-10)
         * 
         * @param utilization Utilization percentage (0-100)
         * @return Sensitivity value (0-10)
         */
        private int convertUtilizationToSensitivity(double utilization) {
            // Apply non-linear transformation to emphasize high utilization
            double normalized = Math.min(100.0, utilization) / 100.0;
            double sensitivity = 10.0 * Math.pow(normalized, 1.5);
            
            // Round to nearest integer and ensure bounds
            return Math.max(1, Math.min(10, (int) Math.round(sensitivity)));
        }
    }
    
    /**
     * Helper class for sorting profile similarities
     */
    private static class ProfileSimilarity {
        public final String jobId;
        public final double similarity;
        
        public ProfileSimilarity(String jobId, double similarity) {
            this.jobId = jobId;
            this.similarity = similarity;
        }
    }
}