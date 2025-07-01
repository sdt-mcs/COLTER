package com.perphproctor.common;

/**
 * Constants used throughout the PerphProctor system.
 * This class contains all constant values to centralize configuration
 * and make the system more maintainable.
 */
public class Constants {
    
    // System configuration
    public static final String PERPHPROCTOR_VERSION = "1.0.0";
    public static final int DEFAULT_MONITORING_INTERVAL_MS = 1000; // 1 second
    public static final int DEFAULT_MODEL_UPDATE_INTERVAL_MS = 60000; // 1 minute
    
    // Resource types
    public static final String RESOURCE_CPU = "cpu";
    public static final String RESOURCE_MEMORY = "memory";
    public static final String RESOURCE_LLC = "llc"; // Last-Level Cache
    public static final String RESOURCE_MBW = "mbw"; // Memory Bandwidth
    public static final String RESOURCE_IO = "io";   // I/O Throughput
    
    // Workload types
    public static final String WORKLOAD_BATCH = "batch";
    public static final String WORKLOAD_LRA = "lra"; // Long-Running Application
    
    // Batch job sensitivity types
    public static final String SENSITIVITY_CPU = "cpu";
    public static final String SENSITIVITY_IO = "io";
    public static final String SENSITIVITY_HYBRID = "hybrid";
    
    // Recommendation score thresholds
    public static final double DEFAULT_RECOMMENDATION_THRESHOLD = 0.5;
    public static final double HIGH_RECOMMENDATION_THRESHOLD = 0.8;
    public static final double LOW_RECOMMENDATION_THRESHOLD = 0.2;
    
    // Continual learning parameters
    public static final int DEFAULT_TREE_COUNT = 100;
    public static final int MIN_SAMPLES_FOR_SPLIT = 5;
    public static final double MAX_RSS_THRESHOLD = 0.05;
    public static final int DEFAULT_MODEL_FRESHNESS_THRESHOLD_MS = 3600000; // 1 hour
    public static final int MAX_UPDATES_PER_TREE = 3;
    
    // QoS parameters
    public static final double DEFAULT_SENSITIVITY_PARAMETER_K = 1.0;
    public static final int MAX_TASK_RETRY_COUNT = 1;
    
    // Performance thresholds
    public static final double QOS_VIOLATION_THRESHOLD = 1.5; // 50% increase in latency
    
    // File paths
    public static final String DEFAULT_MODEL_STORAGE_PATH = "/tmp/perphproctor/models/";
    public static final String DEFAULT_CONFIG_FILE = "perphproctor.properties";
    
    // Feedback mechanisms
    public static final int PREDICTION_FEEDBACK_WINDOW_SIZE = 100;
    public static final double PREDICTION_ERROR_THRESHOLD = 0.2;
    
    private Constants() {
        // Private constructor to prevent instantiation
    }
}
