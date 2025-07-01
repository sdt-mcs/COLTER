package com.perphproctor.common;

import java.util.HashMap;
import java.util.Map;

/**
 * Metrics class to store and manipulate performance metrics
 * collected from various sources in a standardized format.
 */
public class Metrics {
    
    // Resource utilization metrics
    private double cpuUtilization;
    private double memoryUtilization;
    private double llcUtilization;
    private double memoryBandwidth;
    private double ioThroughput;
    
    // Application-specific metrics
    private double appLoad;
    private String batchType;
    
    // Node-specific metrics 
    private double primaryResourceLoad;
    
    // Performance metrics
    private double latency;
    private double throughput;
    
    // Extended metrics for flexible data storage
    private Map<String, Double> additionalMetrics;
    
    // Timestamp for this metrics snapshot
    private long timestamp;
    
    /**
     * Default constructor creating an empty metrics object
     */
    public Metrics() {
        this.additionalMetrics = new HashMap<>();
        this.timestamp = System.currentTimeMillis();
    }
    
    /**
     * Constructor with basic resource utilization parameters
     * 
     * @param cpuUtilization CPU utilization percentage (0-100)
     * @param memoryUtilization Memory utilization percentage (0-100)
     * @param llcUtilization Last-level cache utilization (0-100)
     * @param memoryBandwidth Memory bandwidth usage (MB/s)
     * @param ioThroughput I/O throughput (IOPS)
     */
    public Metrics(double cpuUtilization, double memoryUtilization, 
                  double llcUtilization, double memoryBandwidth, 
                  double ioThroughput) {
        this();
        this.cpuUtilization = cpuUtilization;
        this.memoryUtilization = memoryUtilization;
        this.llcUtilization = llcUtilization;
        this.memoryBandwidth = memoryBandwidth;
        this.ioThroughput = ioThroughput;
    }
    
    /**
     * Constructs a metrics vector from this object for use in prediction models
     * 
     * @return Array of double values representing the feature vector
     */
    public double[] toFeatureVector() {
        return new double[] {
            cpuUtilization,
            memoryUtilization,
            llcUtilization,
            memoryBandwidth,
            ioThroughput,
            appLoad,
            primaryResourceLoad
        };
    }
    
    /**
     * Adds a custom metric value to the additional metrics map
     * 
     * @param key Metric name
     * @param value Metric value
     */
    public void addMetric(String key, double value) {
        additionalMetrics.put(key, value);
    }
    
    /**
     * Gets a custom metric value from the additional metrics map
     * 
     * @param key Metric name
     * @return Metric value, or null if not found
     */
    public Double getMetric(String key) {
        return additionalMetrics.get(key);
    }
    
    /**
     * Merges another metrics object into this one
     * 
     * @param other Metrics object to merge
     */
    public void merge(Metrics other) {
        this.additionalMetrics.putAll(other.additionalMetrics);
        
        // Only overwrite non-zero values
        if (other.cpuUtilization > 0) this.cpuUtilization = other.cpuUtilization;
        if (other.memoryUtilization > 0) this.memoryUtilization = other.memoryUtilization;
        if (other.llcUtilization > 0) this.llcUtilization = other.llcUtilization;
        if (other.memoryBandwidth > 0) this.memoryBandwidth = other.memoryBandwidth;
        if (other.ioThroughput > 0) this.ioThroughput = other.ioThroughput;
        if (other.appLoad > 0) this.appLoad = other.appLoad;
        if (other.latency > 0) this.latency = other.latency;
        if (other.throughput > 0) this.throughput = other.throughput;
        if (other.primaryResourceLoad > 0) this.primaryResourceLoad = other.primaryResourceLoad;
        
        if (other.batchType != null) this.batchType = other.batchType;
        
        // Take the most recent timestamp
        this.timestamp = Math.max(this.timestamp, other.timestamp);
    }
    
    // Getters and setters
    
    public double getCpuUtilization() {
        return cpuUtilization;
    }
    
    public void setCpuUtilization(double cpuUtilization) {
        this.cpuUtilization = cpuUtilization;
    }
    
    public double getMemoryUtilization() {
        return memoryUtilization;
    }
    
    public void setMemoryUtilization(double memoryUtilization) {
        this.memoryUtilization = memoryUtilization;
    }
    
    public double getLlcUtilization() {
        return llcUtilization;
    }
    
    public void setLlcUtilization(double llcUtilization) {
        this.llcUtilization = llcUtilization;
    }
    
    public double getMemoryBandwidth() {
        return memoryBandwidth;
    }
    
    public void setMemoryBandwidth(double memoryBandwidth) {
        this.memoryBandwidth = memoryBandwidth;
    }
    
    public double getIoThroughput() {
        return ioThroughput;
    }
    
    public void setIoThroughput(double ioThroughput) {
        this.ioThroughput = ioThroughput;
    }
    
    public double getAppLoad() {
        return appLoad;
    }
    
    public void setAppLoad(double appLoad) {
        this.appLoad = appLoad;
    }
    
    public String getBatchType() {
        return batchType;
    }
    
    public void setBatchType(String batchType) {
        this.batchType = batchType;
    }
    
    public double getPrimaryResourceLoad() {
        return primaryResourceLoad;
    }
    
    public void setPrimaryResourceLoad(double primaryResourceLoad) {
        this.primaryResourceLoad = primaryResourceLoad;
    }
    
    public double getLatency() {
        return latency;
    }
    
    public void setLatency(double latency) {
        this.latency = latency;
    }
    
    public double getThroughput() {
        return throughput;
    }
    
    public void setThroughput(double throughput) {
        this.throughput = throughput;
    }
    
    public long getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }
    
    public Map<String, Double> getAdditionalMetrics() {
        return additionalMetrics;
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Metrics[");
        sb.append("cpu=").append(cpuUtilization);
        sb.append(", mem=").append(memoryUtilization);
        sb.append(", llc=").append(llcUtilization);
        sb.append(", mbw=").append(memoryBandwidth);
        sb.append(", io=").append(ioThroughput);
        sb.append(", appLoad=").append(appLoad);
        sb.append(", primaryLoad=").append(primaryResourceLoad);
        sb.append(", latency=").append(latency);
        sb.append(", throughput=").append(throughput);
        if (batchType != null) {
            sb.append(", batchType=").append(batchType);
        }
        if (!additionalMetrics.isEmpty()) {
            sb.append(", additionalMetrics=").append(additionalMetrics);
        }
        sb.append(", timestamp=").append(timestamp);
        sb.append("]");
        return sb.toString();
    }
}
