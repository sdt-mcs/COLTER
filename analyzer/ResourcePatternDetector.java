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
 * ResourcePatternDetector analyzes the resource utilization patterns of workloads
 * to classify them into appropriate batch job types. It processes resource metrics
 * collected over time to determine the predominant resource usage characteristics.
 * 
 * This class implements the resource pattern detection functionality as described
 * in the Analyzer component of the PerphProctor framework.
 */
public class ResourcePatternDetector implements Serializable {
    
    private static final long serialVersionUID = 1L;
    private static final Logger LOG = LoggerFactory.getLogger(ResourcePatternDetector.class);
    
    // Thresholds for classification
    private static final double HIGH_CPU_THRESHOLD = 70.0;
    private static final double HIGH_IO_THRESHOLD = 70.0;
    private static final double HIGH_MEMORY_THRESHOLD = 70.0;
    private static final double MODERATE_CPU_THRESHOLD = 50.0;
    private static final double MODERATE_IO_THRESHOLD = 50.0;
    private static final double LOW_CPU_THRESHOLD = 40.0;
    private static final double LOW_IO_THRESHOLD = 40.0;
    private static final double LOW_MEMORY_THRESHOLD = 60.0;
    
    // Minimum confidence threshold for classification
    private static final double MIN_CONFIDENCE_THRESHOLD = 0.6;
    
    /**
     * Constructor
     */
    public ResourcePatternDetector() {
        LOG.info("ResourcePatternDetector initialized");
    }
    
    /**
     * Detect the batch job type based on resource utilization metrics
     * 
     * @param metrics List of collected metrics samples
     * @return Detected batch job type
     */
    public BatchJobType detectBatchType(List<Metrics> metrics) {
        if (metrics == null || metrics.isEmpty()) {
            LOG.warn("Cannot detect batch type: No metrics provided");
            return BatchJobType.UNKNOWN;
        }
        
        // Calculate average utilization across all samples
        Map<String, Double> avgUtilization = calculateAverageUtilization(metrics);
        
        LOG.debug("Average utilization: CPU={}, Memory={}, LLC={}, MBW={}, IO={}",
                 avgUtilization.get("cpu"),
                 avgUtilization.get("memory"),
                 avgUtilization.get("llc"),
                 avgUtilization.get("mbw"),
                 avgUtilization.get("io"));
        
        return detectBatchType(avgUtilization);
    }
    
    /**
     * Detect batch job type based on resource utilization map
     * 
     * @param resourceUtilization Map of resource names to utilization values
     * @return Detected batch job type
     */
    public BatchJobType detectBatchType(Map<String, Double> resourceUtilization) {
        if (resourceUtilization == null || resourceUtilization.isEmpty()) {
            LOG.warn("Cannot detect batch type: No resource utilization data provided");
            return BatchJobType.UNKNOWN;
        }
        
        double cpuUtil = resourceUtilization.getOrDefault("cpu", 0.0);
        double memoryUtil = resourceUtilization.getOrDefault("memory", 0.0);
        double llcUtil = resourceUtilization.getOrDefault("llc", 0.0);
        double mbwUtil = resourceUtilization.getOrDefault("mbw", 0.0);
        double ioUtil = resourceUtilization.getOrDefault("io", 0.0);
        
        // Calculate confidence scores for each batch type
        Map<BatchJobType, Double> confidenceScores = new HashMap<>();
        
        // CPU-intensive confidence
        double cpuIntensiveScore = calculateCpuIntensiveConfidence(cpuUtil, memoryUtil, ioUtil, llcUtil);
        confidenceScores.put(BatchJobType.CPU_INTENSIVE, cpuIntensiveScore);
        
        // IO-intensive confidence
        double ioIntensiveScore = calculateIoIntensiveConfidence(cpuUtil, ioUtil, mbwUtil);
        confidenceScores.put(BatchJobType.IO_INTENSIVE, ioIntensiveScore);
        
        // Memory-intensive confidence
        double memoryIntensiveScore = calculateMemoryIntensiveConfidence(cpuUtil, memoryUtil, ioUtil, llcUtil, mbwUtil);
        confidenceScores.put(BatchJobType.MEMORY_INTENSIVE, memoryIntensiveScore);
        
        // Hybrid confidence
        double hybridScore = calculateHybridConfidence(cpuUtil, ioUtil, memoryUtil);
        confidenceScores.put(BatchJobType.HYBRID, hybridScore);
        
        // Find batch type with highest confidence
        BatchJobType detectedType = BatchJobType.UNKNOWN;
        double maxConfidence = 0.0;
        
        for (Map.Entry<BatchJobType, Double> entry : confidenceScores.entrySet()) {
            if (entry.getValue() > maxConfidence) {
                maxConfidence = entry.getValue();
                detectedType = entry.getKey();
            }
        }
        
        // If confidence is too low, return UNKNOWN
        if (maxConfidence < MIN_CONFIDENCE_THRESHOLD) {
            LOG.info("Insufficient confidence for classification (max confidence: {}), returning UNKNOWN", 
                    maxConfidence);
            return BatchJobType.UNKNOWN;
        }
        
        LOG.info("Detected batch type {} with confidence {}", detectedType, maxConfidence);
        LOG.debug("Confidence scores: CPU_INTENSIVE={}, IO_INTENSIVE={}, MEMORY_INTENSIVE={}, HYBRID={}",
                 confidenceScores.get(BatchJobType.CPU_INTENSIVE),
                 confidenceScores.get(BatchJobType.IO_INTENSIVE),
                 confidenceScores.get(BatchJobType.MEMORY_INTENSIVE),
                 confidenceScores.get(BatchJobType.HYBRID));
        
        return detectedType;
    }
    
    /**
     * Calculate average utilization across multiple metrics samples
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
            // Approx. normalize to a [0,100] scale vs. platform peak memory bandwidth
            // (~115 GB/s/socket), aligning MBW with CPU/LLC percentages for batch-type classification.
            totalMbw += sample.getMemoryBandwidth() / 1000.0;
            // Approx. normalize to a [0,100] scale vs. platform I/O baseline (~500 MB/s/node).
            totalIo += sample.getIoThroughput() / 5.0;
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
     * Calculate confidence score for CPU-intensive classification
     * 
     * @param cpuUtil CPU utilization percentage
     * @param memoryUtil Memory utilization percentage
     * @param ioUtil IO utilization percentage
     * @param llcUtil Last-level cache utilization percentage
     * @return Confidence score (0-1)
     */
    private double calculateCpuIntensiveConfidence(double cpuUtil, double memoryUtil, 
                                                  double ioUtil, double llcUtil) {
        double confidence = 0.0;
        
        // Basic CPU-intensive criteria
        if (cpuUtil > HIGH_CPU_THRESHOLD && ioUtil < LOW_IO_THRESHOLD) {
            confidence = 0.8;
        } else if (cpuUtil > MODERATE_CPU_THRESHOLD && ioUtil < LOW_IO_THRESHOLD) {
            confidence = 0.6;
        } else if (cpuUtil > MODERATE_CPU_THRESHOLD) {
            confidence = 0.4;
        } else {
            confidence = 0.2;
        }
        
        // Adjust based on LLC utilization
        if (llcUtil > 60.0) {
            confidence += 0.1;
        }
        
        // Penalize for high IO
        if (ioUtil > MODERATE_IO_THRESHOLD) {
            confidence -= 0.2;
        }
        
        // Ensure confidence is within [0,1]
        return Math.max(0.0, Math.min(1.0, confidence));
    }
    
    /**
     * Calculate confidence score for IO-intensive classification
     * 
     * @param cpuUtil CPU utilization percentage
     * @param ioUtil IO utilization percentage
     * @param mbwUtil Memory bandwidth utilization percentage
     * @return Confidence score (0-1)
     */
    private double calculateIoIntensiveConfidence(double cpuUtil, double ioUtil, double mbwUtil) {
        double confidence = 0.0;
        
        // Basic IO-intensive criteria
        if (ioUtil > HIGH_IO_THRESHOLD && cpuUtil < LOW_CPU_THRESHOLD) {
            confidence = 0.8;
        } else if (ioUtil > MODERATE_IO_THRESHOLD && cpuUtil < LOW_CPU_THRESHOLD) {
            confidence = 0.6;
        } else if (ioUtil > MODERATE_IO_THRESHOLD) {
            confidence = 0.4;
        } else {
            confidence = 0.2;
        }
        
        // Adjust based on memory bandwidth
        if (mbwUtil > 50.0) {
            confidence += 0.1;
        }
        
        // Penalize for high CPU
        if (cpuUtil > MODERATE_CPU_THRESHOLD) {
            confidence -= 0.2;
        }
        
        // Ensure confidence is within [0,1]
        return Math.max(0.0, Math.min(1.0, confidence));
    }
    
    /**
     * Calculate confidence score for memory-intensive classification
     * 
     * @param cpuUtil CPU utilization percentage
     * @param memoryUtil Memory utilization percentage
     * @param ioUtil IO utilization percentage
     * @param llcUtil Last-level cache utilization percentage
     * @param mbwUtil Memory bandwidth utilization percentage
     * @return Confidence score (0-1)
     */
    private double calculateMemoryIntensiveConfidence(double cpuUtil, double memoryUtil, 
                                                    double ioUtil, double llcUtil, double mbwUtil) {
        double confidence = 0.0;
        
        // Basic memory-intensive criteria
        if (memoryUtil > HIGH_MEMORY_THRESHOLD && cpuUtil < LOW_MEMORY_THRESHOLD && ioUtil < LOW_IO_THRESHOLD) {
            confidence = 0.8;
        } else if (memoryUtil > MODERATE_CPU_THRESHOLD && cpuUtil < LOW_MEMORY_THRESHOLD) {
            confidence = 0.6;
        } else if (memoryUtil > MODERATE_CPU_THRESHOLD) {
            confidence = 0.4;
        } else {
            confidence = 0.2;
        }
        
        // Adjust based on memory bandwidth and LLC
        if (mbwUtil > 60.0) {
            confidence += 0.1;
        }
        
        if (llcUtil > 50.0) {
            confidence += 0.1;
        }
        
        // Penalize for high CPU and IO
        if (cpuUtil > MODERATE_CPU_THRESHOLD) {
            confidence -= 0.1;
        }
        
        if (ioUtil > MODERATE_IO_THRESHOLD) {
            confidence -= 0.1;
        }
        
        // Ensure confidence is within [0,1]
        return Math.max(0.0, Math.min(1.0, confidence));
    }
    
    /**
     * Calculate confidence score for hybrid classification
     * 
     * @param cpuUtil CPU utilization percentage
     * @param ioUtil IO utilization percentage
     * @param memoryUtil Memory utilization percentage
     * @return Confidence score (0-1)
     */
    private double calculateHybridConfidence(double cpuUtil, double ioUtil, double memoryUtil) {
        double confidence = 0.0;
        
        // Basic hybrid criteria
        if (cpuUtil > MODERATE_CPU_THRESHOLD && ioUtil > MODERATE_IO_THRESHOLD) {
            confidence = 0.8;
        } else if ((cpuUtil > MODERATE_CPU_THRESHOLD && ioUtil > LOW_IO_THRESHOLD) ||
                  (ioUtil > MODERATE_IO_THRESHOLD && cpuUtil > LOW_CPU_THRESHOLD)) {
            confidence = 0.6;
        } else if (cpuUtil > LOW_CPU_THRESHOLD && ioUtil > LOW_IO_THRESHOLD) {
            confidence = 0.4;
        } else {
            confidence = 0.2;
        }
        
        // Adjust based on balanced resource usage
        double cpuIoDiff = Math.abs(cpuUtil - ioUtil);
        if (cpuIoDiff < 15.0) {
            confidence += 0.1; // More balanced usage indicates hybrid
        }
        
        // Also consider memory
        if (memoryUtil > MODERATE_CPU_THRESHOLD) {
            confidence += 0.1;
        }
        
        // Ensure confidence is within [0,1]
        return Math.max(0.0, Math.min(1.0, confidence));
    }
    
    /**
     * Analyze time-series patterns in resource utilization
     * 
     * @param metrics List of metrics samples in time order
     * @return Map of resource names to pattern information
     */
    public Map<String, Object> analyzeTimeSeriesPatterns(List<Metrics> metrics) {
        Map<String, Object> patterns = new HashMap<>();
        
        if (metrics == null || metrics.size() < 5) {
            LOG.warn("Cannot analyze time series: Insufficient data points");
            return patterns;
        }
        
        // Analyze trends for each resource
        patterns.put("cpuTrend", calculateTrend(extractMetricSeries(metrics, "cpu")));
        patterns.put("memoryTrend", calculateTrend(extractMetricSeries(metrics, "memory")));
        patterns.put("ioTrend", calculateTrend(extractMetricSeries(metrics, "io")));
        
        // Analyze stability
        patterns.put("cpuStability", calculateStability(extractMetricSeries(metrics, "cpu")));
        patterns.put("memoryStability", calculateStability(extractMetricSeries(metrics, "memory")));
        patterns.put("ioStability", calculateStability(extractMetricSeries(metrics, "io")));
        
        // Analyze periodicity
        patterns.put("hasPeriodicity", detectPeriodicity(metrics));
        
        LOG.debug("Time series analysis: {}", patterns);
        
        return patterns;
    }
    
    /**
     * Extract a single metric series from metrics list
     * 
     * @param metrics List of metrics samples
     * @param metricName Name of metric to extract
     * @return Array of metric values
     */
    private double[] extractMetricSeries(List<Metrics> metrics, String metricName) {
        double[] series = new double[metrics.size()];
        
        for (int i = 0; i < metrics.size(); i++) {
            Metrics sample = metrics.get(i);
            
            switch (metricName) {
                case "cpu":
                    series[i] = sample.getCpuUtilization();
                    break;
                case "memory":
                    series[i] = sample.getMemoryUtilization();
                    break;
                case "llc":
                    series[i] = sample.getLlcUtilization();
                    break;
                case "mbw":
                    series[i] = sample.getMemoryBandwidth();
                    break;
                case "io":
                    series[i] = sample.getIoThroughput();
                    break;
                default:
                    series[i] = 0.0;
            }
        }
        
        return series;
    }
    
    /**
     * Calculate trend of a time series (positive = increasing, negative = decreasing)
     * 
     * @param series Array of metric values
     * @return Trend value
     */
    private double calculateTrend(double[] series) {
        if (series.length < 2) {
            return 0.0;
        }
        
        // Simple linear regression
        int n = series.length;
        double sumX = 0.0;
        double sumY = 0.0;
        double sumXY = 0.0;
        double sumX2 = 0.0;
        
        for (int i = 0; i < n; i++) {
            sumX += i;
            sumY += series[i];
            sumXY += i * series[i];
            sumX2 += i * i;
        }
        
        double denominator = n * sumX2 - sumX * sumX;
        if (Math.abs(denominator) < 1e-10) {
            return 0.0; // Avoid division by zero
        }
        
        // Calculate slope
        return (n * sumXY - sumX * sumY) / denominator;
    }
    
    /**
     * Calculate stability of a time series (higher = more stable)
     * 
     * @param series Array of metric values
     * @return Stability value (0-1)
     */
    private double calculateStability(double[] series) {
        if (series.length < 2) {
            return 1.0;
        }
        
        // Calculate mean
        double sum = 0.0;
        for (double value : series) {
            sum += value;
        }
        double mean = sum / series.length;
        
        // Calculate standard deviation
        double sumSquaredDiff = 0.0;
        for (double value : series) {
            double diff = value - mean;
            sumSquaredDiff += diff * diff;
        }
        double stdDev = Math.sqrt(sumSquaredDiff / series.length);
        
        // Calculate coefficient of variation (CV)
        double cv = mean == 0 ? 0.0 : stdDev / mean;
        
        // Convert CV to stability (0-1)
        return Math.max(0.0, Math.min(1.0, 1.0 - cv));
    }
    
    /**
     * Detect periodicity in metrics
     * 
     * @param metrics List of metrics samples
     * @return true if periodicity detected
     */
    private boolean detectPeriodicity(List<Metrics> metrics) {
        // Simplified periodicity detection
        // In a real implementation, this would use autocorrelation or Fourier analysis
        
        // For now, return false (no periodicity detected)
        return false;
    }
    
    /**
     * Get detailed classification for all batch job types
     * 
     * @param metrics List of metrics samples
     * @return Map of batch job types to confidence scores
     */
    public Map<BatchJobType, Double> getDetailedClassification(List<Metrics> metrics) {
        if (metrics == null || metrics.isEmpty()) {
            return new HashMap<>();
        }
        
        // Calculate average utilization
        Map<String, Double> avgUtilization = calculateAverageUtilization(metrics);
        
        double cpuUtil = avgUtilization.getOrDefault("cpu", 0.0);
        double memoryUtil = avgUtilization.getOrDefault("memory", 0.0);
        double llcUtil = avgUtilization.getOrDefault("llc", 0.0);
        double mbwUtil = avgUtilization.getOrDefault("mbw", 0.0);
        double ioUtil = avgUtilization.getOrDefault("io", 0.0);
        
        // Calculate confidence scores for each batch type
        Map<BatchJobType, Double> confidenceScores = new HashMap<>();
        
        confidenceScores.put(BatchJobType.CPU_INTENSIVE, 
                calculateCpuIntensiveConfidence(cpuUtil, memoryUtil, ioUtil, llcUtil));
        
        confidenceScores.put(BatchJobType.IO_INTENSIVE, 
                calculateIoIntensiveConfidence(cpuUtil, ioUtil, mbwUtil));
        
        confidenceScores.put(BatchJobType.MEMORY_INTENSIVE, 
                calculateMemoryIntensiveConfidence(cpuUtil, memoryUtil, ioUtil, llcUtil, mbwUtil));
        
        confidenceScores.put(BatchJobType.HYBRID, 
                calculateHybridConfidence(cpuUtil, ioUtil, memoryUtil));
        
        // Add UNKNOWN classification with a baseline confidence
        confidenceScores.put(BatchJobType.UNKNOWN, 0.1);
        
        return confidenceScores;
    }
}