package com.perphproctor.analyzer;

import com.perphproctor.common.Metrics;
import com.perphproctor.common.WorkloadTypes.BatchJobType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * SensitivityClassifier analyzes workload metrics to determine their sensitivity
 * to different resource types. This information is used to guide scheduling decisions
 * that minimize interference between co-located workloads.
 * 
 * This class quantifies how sensitive a workload is to CPU, memory, LLC, memory bandwidth, 
 * and I/O resources on a scale of 0-10, where higher values indicate greater sensitivity.
 */
public class SensitivityClassifier implements Serializable {
    
    private static final long serialVersionUID = 1L;
    private static final Logger LOG = LoggerFactory.getLogger(SensitivityClassifier.class);
    
    // Sensitivity calculation parameters
    private static final double UTILIZATION_WEIGHT = 0.6;
    private static final double VARIANCE_WEIGHT = 0.4;
    private static final double HIGH_UTILIZATION_THRESHOLD = 70.0;
    private static final double MODERATE_UTILIZATION_THRESHOLD = 40.0;
    
    /**
     * Constructor
     */
    public SensitivityClassifier() {
        LOG.info("SensitivityClassifier initialized");
    }
    
    /**
     * Classify resource sensitivity based on metrics samples
     * 
     * @param metrics List of metrics samples
     * @param batchType Detected batch job type
     * @return Map of resource names to sensitivity values (0-10)
     */
    public Map<String, Integer> classifySensitivity(List<Metrics> metrics, BatchJobType batchType) {
        if (metrics == null || metrics.isEmpty()) {
            LOG.warn("Cannot classify sensitivity: No metrics provided");
            return getDefaultSensitivity(batchType);
        }
        
        // Calculate resource utilization statistics
        Map<String, Double> avgUtilization = calculateAverageUtilization(metrics);
        Map<String, Double> varianceUtilization = calculateVarianceUtilization(metrics, avgUtilization);
        
        LOG.debug("Average utilization: {}", avgUtilization);
        LOG.debug("Variance utilization: {}", varianceUtilization);
        
        return classifySensitivity(avgUtilization, varianceUtilization, batchType);
    }
    
    /**
     * Classify resource sensitivity based on utilization statistics
     * 
     * @param avgUtilization Map of resource names to average utilization values
     * @param varianceUtilization Map of resource names to utilization variance values
     * @param batchType Detected batch job type
     * @return Map of resource names to sensitivity values (0-10)
     */
    public Map<String, Integer> classifySensitivity(
            Map<String, Double> avgUtilization, 
            Map<String, Double> varianceUtilization,
            BatchJobType batchType) {
        
        // Start with default values based on batch type
        Map<String, Integer> sensitivity = getDefaultSensitivity(batchType);
        
        if (avgUtilization == null || avgUtilization.isEmpty()) {
            return sensitivity;
        }
        
        // Adjust sensitivity based on observed utilization and variance
        adjustCpuSensitivity(sensitivity, avgUtilization, varianceUtilization);
        adjustMemorySensitivity(sensitivity, avgUtilization, varianceUtilization);
        adjustLlcSensitivity(sensitivity, avgUtilization, varianceUtilization);
        adjustMbwSensitivity(sensitivity, avgUtilization, varianceUtilization);
        adjustIoSensitivity(sensitivity, avgUtilization, varianceUtilization);
        
        LOG.debug("Final sensitivity classification: {}", sensitivity);
        
        return sensitivity;
    }
    
    /**
     * Classify resource sensitivity based on average utilization
     * 
     * @param avgUtilization Map of resource names to average utilization values
     * @param batchType Detected batch job type
     * @return Map of resource names to sensitivity values (0-10)
     */
    public Map<String, Integer> classifySensitivity(
            Map<String, Double> avgUtilization, BatchJobType batchType) {
        
        // Create empty variance map
        Map<String, Double> emptyVariance = new HashMap<>();
        for (String resource : avgUtilization.keySet()) {
            emptyVariance.put(resource, 0.0);
        }
        
        // Use only utilization for classification
        return classifySensitivity(avgUtilization, emptyVariance, batchType);
    }
    
    /**
     * Get default sensitivity values based on batch type
     * 
     * @param batchType Batch job type
     * @return Map of resource names to default sensitivity values
     */
    private Map<String, Integer> getDefaultSensitivity(BatchJobType batchType) {
        Map<String, Integer> sensitivity = new HashMap<>();
        
        // Default moderate sensitivity for all resources
        sensitivity.put("cpu", 5);
        sensitivity.put("memory", 5);
        sensitivity.put("llc", 5);
        sensitivity.put("mbw", 5);
        sensitivity.put("io", 5);
        
        // Adjust based on batch type
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
                sensitivity.put("llc", 7);
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
                
            case UNKNOWN:
            default:
                // Keep default moderate sensitivity
                break;
        }
        
        return sensitivity;
    }
    
    /**
     * Calculate average utilization for multiple metrics samples
     * 
     * @param metrics List of metrics samples
     * @return Map of resource names to average utilization values
     */
    private Map<String, Double> calculateAverageUtilization(List<Metrics> metrics) {
        Map<String, Double> avgUtilization = new HashMap<>();
        
        if (metrics.isEmpty()) {
            return avgUtilization;
        }
        
        double totalCpu = 0.0;
        double totalMemory = 0.0;
        double totalLlc = 0.0;
        double totalMbw = 0.0;
        double totalIo = 0.0;
        
        for (Metrics sample : metrics) {
            totalCpu += sample.getCpuUtilization();
            totalMemory += sample.getMemoryUtilization();
            totalLlc += sample.getLlcUtilization();
            totalMbw += sample.getMemoryBandwidth();
            totalIo += sample.getIoThroughput();
        }
        
        int count = metrics.size();
        
        avgUtilization.put("cpu", totalCpu / count);
        avgUtilization.put("memory", totalMemory / count);
        avgUtilization.put("llc", totalLlc / count);
        avgUtilization.put("mbw", totalMbw / count);
        avgUtilization.put("io", totalIo / count);
        
        return avgUtilization;
    }
    
    /**
     * Calculate utilization variance for multiple metrics samples
     * 
     * @param metrics List of metrics samples
     * @param avgUtilization Map of resource names to average utilization values
     * @return Map of resource names to utilization variance values
     */
    private Map<String, Double> calculateVarianceUtilization(
            List<Metrics> metrics, Map<String, Double> avgUtilization) {
        
        Map<String, Double> variance = new HashMap<>();
        
        if (metrics.isEmpty()) {
            return variance;
        }
        
        double sumCpuSquaredDiff = 0.0;
        double sumMemorySquaredDiff = 0.0;
        double sumLlcSquaredDiff = 0.0;
        double sumMbwSquaredDiff = 0.0;
        double sumIoSquaredDiff = 0.0;
        
        double avgCpu = avgUtilization.getOrDefault("cpu", 0.0);
        double avgMemory = avgUtilization.getOrDefault("memory", 0.0);
        double avgLlc = avgUtilization.getOrDefault("llc", 0.0);
        double avgMbw = avgUtilization.getOrDefault("mbw", 0.0);
        double avgIo = avgUtilization.getOrDefault("io", 0.0);
        
        for (Metrics sample : metrics) {
            double cpuDiff = sample.getCpuUtilization() - avgCpu;
            double memoryDiff = sample.getMemoryUtilization() - avgMemory;
            double llcDiff = sample.getLlcUtilization() - avgLlc;
            double mbwDiff = sample.getMemoryBandwidth() - avgMbw;
            double ioDiff = sample.getIoThroughput() - avgIo;
            
            sumCpuSquaredDiff += cpuDiff * cpuDiff;
            sumMemorySquaredDiff += memoryDiff * memoryDiff;
            sumLlcSquaredDiff += llcDiff * llcDiff;
            sumMbwSquaredDiff += mbwDiff * mbwDiff;
            sumIoSquaredDiff += ioDiff * ioDiff;
        }
        
        int count = metrics.size();
        
        variance.put("cpu", sumCpuSquaredDiff / count);
        variance.put("memory", sumMemorySquaredDiff / count);
        variance.put("llc", sumLlcSquaredDiff / count);
        variance.put("mbw", sumMbwSquaredDiff / count);
        variance.put("io", sumIoSquaredDiff / count);
        
        return variance;
    }
    
    /**
     * Adjust CPU sensitivity based on utilization data
     * 
     * @param sensitivity Map of resource sensitivities to adjust
     * @param avgUtilization Average utilization values
     * @param varianceUtilization Utilization variance values
     */
    private void adjustCpuSensitivity(
            Map<String, Integer> sensitivity,
            Map<String, Double> avgUtilization,
            Map<String, Double> varianceUtilization) {
        
        double cpuUtil = avgUtilization.getOrDefault("cpu", 0.0);
        double cpuVar = varianceUtilization.getOrDefault("cpu", 0.0);
        
        // Normalize variance to a 0-10 scale
        double normalizedVar = Math.min(10.0, cpuVar / 100.0);
        
        // Calculate sensitivity score from utilization and variance
        double utilizationScore = calculateUtilizationSensitivity(cpuUtil);
        double varianceScore = 10.0 - normalizedVar; // Lower variance means higher sensitivity
        
        // Weighted score
        double score = UTILIZATION_WEIGHT * utilizationScore + VARIANCE_WEIGHT * varianceScore;
        
        // Round to nearest integer and update
        sensitivity.put("cpu", Math.max(1, Math.min(10, (int) Math.round(score))));
    }
    
    /**
     * Adjust memory sensitivity based on utilization data
     * 
     * @param sensitivity Map of resource sensitivities to adjust
     * @param avgUtilization Average utilization values
     * @param varianceUtilization Utilization variance values
     */
    private void adjustMemorySensitivity(
            Map<String, Integer> sensitivity,
            Map<String, Double> avgUtilization,
            Map<String, Double> varianceUtilization) {
        
        double memoryUtil = avgUtilization.getOrDefault("memory", 0.0);
        double memoryVar = varianceUtilization.getOrDefault("memory", 0.0);
        
        // Normalize variance to a 0-10 scale
        double normalizedVar = Math.min(10.0, memoryVar / 100.0);
        
        // Calculate sensitivity score from utilization and variance
        double utilizationScore = calculateUtilizationSensitivity(memoryUtil);
        double varianceScore = 10.0 - normalizedVar; // Lower variance means higher sensitivity
        
        // Memory is particularly sensitive to low variance (steady usage)
        double score = UTILIZATION_WEIGHT * utilizationScore + 
                      (VARIANCE_WEIGHT + 0.1) * varianceScore;
        
        // Round to nearest integer and update
        sensitivity.put("memory", Math.max(1, Math.min(10, (int) Math.round(score))));
    }
    
    /**
     * Adjust LLC sensitivity based on utilization data
     * 
     * @param sensitivity Map of resource sensitivities to adjust
     * @param avgUtilization Average utilization values
     * @param varianceUtilization Utilization variance values
     */
    private void adjustLlcSensitivity(
            Map<String, Integer> sensitivity,
            Map<String, Double> avgUtilization,
            Map<String, Double> varianceUtilization) {
        
        double llcUtil = avgUtilization.getOrDefault("llc", 0.0);
        double llcVar = varianceUtilization.getOrDefault("llc", 0.0);
        double cpuUtil = avgUtilization.getOrDefault("cpu", 0.0);
        
        // Normalize variance to a 0-10 scale
        double normalizedVar = Math.min(10.0, llcVar / 100.0);
        
        // Calculate sensitivity score from utilization and variance
        double utilizationScore = calculateUtilizationSensitivity(llcUtil);
        double varianceScore = 10.0 - normalizedVar; // Lower variance means higher sensitivity
        
        // LLC sensitivity is related to CPU utilization
        double cpuFactor = Math.min(1.0, cpuUtil / 70.0);
        double score = (UTILIZATION_WEIGHT * utilizationScore + 
                      VARIANCE_WEIGHT * varianceScore) * (0.7 + 0.3 * cpuFactor);
        
        // Round to nearest integer and update
        sensitivity.put("llc", Math.max(1, Math.min(10, (int) Math.round(score))));
    }
    
    /**
     * Adjust memory bandwidth sensitivity based on utilization data
     * 
     * @param sensitivity Map of resource sensitivities to adjust
     * @param avgUtilization Average utilization values
     * @param varianceUtilization Utilization variance values
     */
    private void adjustMbwSensitivity(
            Map<String, Integer> sensitivity,
            Map<String, Double> avgUtilization,
            Map<String, Double> varianceUtilization) {
        
        double mbwUtil = avgUtilization.getOrDefault("mbw", 0.0);
        double mbwVar = varianceUtilization.getOrDefault("mbw", 0.0);
        double memoryUtil = avgUtilization.getOrDefault("memory", 0.0);
        
        // Normalize MBW to percentage scale
        double normalizedMbwUtil = Math.min(100.0, mbwUtil / 100.0);
        
        // Normalize variance to a 0-10 scale
        double normalizedVar = Math.min(10.0, mbwVar / 1000.0);
        
        // Calculate sensitivity score from utilization and variance
        double utilizationScore = calculateUtilizationSensitivity(normalizedMbwUtil);
        double varianceScore = 10.0 - normalizedVar; // Lower variance means higher sensitivity
        
        // MBW sensitivity is related to memory utilization
        double memoryFactor = Math.min(1.0, memoryUtil / 70.0);
        double score = (UTILIZATION_WEIGHT * utilizationScore + 
                      VARIANCE_WEIGHT * varianceScore) * (0.7 + 0.3 * memoryFactor);
        
        // Round to nearest integer and update
        sensitivity.put("mbw", Math.max(1, Math.min(10, (int) Math.round(score))));
    }
    
    /**
     * Adjust I/O sensitivity based on utilization data
     * 
     * @param sensitivity Map of resource sensitivities to adjust
     * @param avgUtilization Average utilization values
     * @param varianceUtilization Utilization variance values
     */
    private void adjustIoSensitivity(
            Map<String, Integer> sensitivity,
            Map<String, Double> avgUtilization,
            Map<String, Double> varianceUtilization) {
        
        double ioUtil = avgUtilization.getOrDefault("io", 0.0);
        double ioVar = varianceUtilization.getOrDefault("io", 0.0);
        
        // Normalize IO to percentage scale
        double normalizedIoUtil = Math.min(100.0, ioUtil / 50.0);
        
        // Normalize variance to a 0-10 scale
        double normalizedVar = Math.min(10.0, ioVar / 500.0);
        
        // Calculate sensitivity score from utilization and variance
        double utilizationScore = calculateUtilizationSensitivity(normalizedIoUtil);
        
        // For IO, higher variance actually indicates higher sensitivity (bursty IO)
        double varianceScore = Math.min(10.0, normalizedVar);
        
        // Weighted score, with higher weight on utilization for IO
        double score = (UTILIZATION_WEIGHT + 0.1) * utilizationScore + 
                      (VARIANCE_WEIGHT - 0.1) * varianceScore;
        
        // Round to nearest integer and update
        sensitivity.put("io", Math.max(1, Math.min(10, (int) Math.round(score))));
    }
    
    /**
     * Calculate sensitivity score based on resource utilization
     * 
     * @param utilization Resource utilization percentage
     * @return Sensitivity score (0-10)
     */
    private double calculateUtilizationSensitivity(double utilization) {
        if (utilization >= HIGH_UTILIZATION_THRESHOLD) {
            // High utilization indicates high sensitivity (8-10)
            return 8.0 + (utilization - HIGH_UTILIZATION_THRESHOLD) * 2.0 / 
                  (100.0 - HIGH_UTILIZATION_THRESHOLD);
        } else if (utilization >= MODERATE_UTILIZATION_THRESHOLD) {
            // Moderate utilization indicates moderate sensitivity (5-8)
            return 5.0 + (utilization - MODERATE_UTILIZATION_THRESHOLD) * 3.0 / 
                  (HIGH_UTILIZATION_THRESHOLD - MODERATE_UTILIZATION_THRESHOLD);
        } else {
            // Low utilization indicates low sensitivity (1-5)
            return 1.0 + utilization * 4.0 / MODERATE_UTILIZATION_THRESHOLD;
        }
    }
    
    /**
     * Analyze cross-resource interactions and adjust sensitivities
     * 
     * @param sensitivity Map of resource sensitivities to adjust
     * @param avgUtilization Average utilization values
     */
    public void analyzeCrossResourceInteractions(
            Map<String, Integer> sensitivity, 
            Map<String, Double> avgUtilization) {
        
        // CPU-LLC interaction
        double cpuUtil = avgUtilization.getOrDefault("cpu", 0.0);
        double llcUtil = avgUtilization.getOrDefault("llc", 0.0);
        
        // If both CPU and LLC are highly utilized, increase both sensitivities
        if (cpuUtil > 60.0 && llcUtil > 60.0) {
            int cpuSens = sensitivity.getOrDefault("cpu", 5);
            int llcSens = sensitivity.getOrDefault("llc", 5);
            
            sensitivity.put("cpu", Math.min(10, cpuSens + 1));
            sensitivity.put("llc", Math.min(10, llcSens + 1));
        }
        
        // Memory-MBW interaction
        double memoryUtil = avgUtilization.getOrDefault("memory", 0.0);
        double mbwUtil = avgUtilization.getOrDefault("mbw", 0.0) / 100.0; // Normalize
        
        // If memory is highly utilized but MBW is low, increase memory sensitivity
        if (memoryUtil > 70.0 && mbwUtil < 30.0) {
            int memorySens = sensitivity.getOrDefault("memory", 5);
            sensitivity.put("memory", Math.min(10, memorySens + 1));
        }
        
        // IO-CPU interaction for hybrid workloads
        double ioUtil = avgUtilization.getOrDefault("io", 0.0) / 50.0; // Normalize
        
        // If both CPU and IO are moderately utilized, this is likely a hybrid workload
        if (cpuUtil > 50.0 && ioUtil > 50.0) {
            int cpuSens = sensitivity.getOrDefault("cpu", 5);
            int ioSens = sensitivity.getOrDefault("io", 5);
            
            sensitivity.put("cpu", Math.min(10, cpuSens + 1));
            sensitivity.put("io", Math.min(10, ioSens + 1));
        }
    }
    
    /**
     * Detect resource bottlenecks based on utilization patterns
     * 
     * @param metrics List of metrics samples
     * @return Map of resource names to bottleneck likelihood (0-1)
     */
    public Map<String, Double> detectBottlenecks(List<Metrics> metrics) {
        Map<String, Double> bottlenecks = new HashMap<>();
        
        if (metrics == null || metrics.isEmpty()) {
            return bottlenecks;
        }
        
        // Calculate average utilization
        Map<String, Double> avgUtilization = calculateAverageUtilization(metrics);
        
        // Calculate bottleneck likelihood for each resource
        bottlenecks.put("cpu", calculateBottleneckLikelihood(avgUtilization.getOrDefault("cpu", 0.0)));
        bottlenecks.put("memory", calculateBottleneckLikelihood(avgUtilization.getOrDefault("memory", 0.0)));
        bottlenecks.put("llc", calculateBottleneckLikelihood(avgUtilization.getOrDefault("llc", 0.0)));
        bottlenecks.put("mbw", calculateBottleneckLikelihood(avgUtilization.getOrDefault("mbw", 0.0) / 100.0));
        bottlenecks.put("io", calculateBottleneckLikelihood(avgUtilization.getOrDefault("io", 0.0) / 50.0));
        
        LOG.debug("Bottleneck likelihoods: {}", bottlenecks);
        
        return bottlenecks;
    }
    
    /**
     * Calculate bottleneck likelihood based on resource utilization
     * 
     * @param utilization Resource utilization percentage
     * @return Bottleneck likelihood (0-1)
     */
    private double calculateBottleneckLikelihood(double utilization) {
        if (utilization >= 90.0) {
            return 1.0; // Definitely a bottleneck
        } else if (utilization >= 75.0) {
            return 0.7 + (utilization - 75.0) * 0.3 / 15.0; // 0.7-1.0 range
        } else if (utilization >= 60.0) {
            return 0.4 + (utilization - 60.0) * 0.3 / 15.0; // 0.4-0.7 range
        } else if (utilization >= 40.0) {
            return 0.1 + (utilization - 40.0) * 0.3 / 20.0; // 0.1-0.4 range
        } else {
            return Math.max(0.0, utilization / 400.0); // 0.0-0.1 range
        }
    }
}