package com.perphproctor.common;

/**
 * WorkloadTypes class defines the different types of workloads
 * and their characteristics for classification and prediction purposes.
 */
public class WorkloadTypes {
    
    /**
     * Enum representing different types of batch jobs based on their 
     * resource sensitivity characteristics
     */
    public enum BatchJobType {
        CPU_INTENSIVE("CPU", "High CPU utilization, low I/O, moderate memory"),
        IO_INTENSIVE("IO", "High I/O activity, moderate CPU and memory"),
        HYBRID("HYBRID", "Balanced CPU and I/O utilization"),
        MEMORY_INTENSIVE("MEMORY", "High memory utilization, moderate CPU and I/O"),
        UNKNOWN("UNKNOWN", "Resource pattern not yet classified");
        
        private final String code;
        private final String description;
        
        BatchJobType(String code, String description) {
            this.code = code;
            this.description = description;
        }
        
        public String getCode() {
            return code;
        }
        
        public String getDescription() {
            return description;
        }
        
        /**
         * Get a BatchJobType from its code
         * 
         * @param code The code to look up
         * @return The matching BatchJobType or UNKNOWN if not found
         */
        public static BatchJobType fromCode(String code) {
            if (code == null) {
                return UNKNOWN;
            }
            
            for (BatchJobType type : values()) {
                if (type.code.equalsIgnoreCase(code)) {
                    return type;
                }
            }
            return UNKNOWN;
        }
    }
    
    /**
     * Enum representing different types of LRA components
     */
    public enum LraComponentType {
        FRONTEND("FRONTEND", "User interface and request handling"),
        BACKEND("BACKEND", "Business logic processing"),
        DATABASE("DATABASE", "Data storage and retrieval"),
        MIDDLEWARE("MIDDLEWARE", "Communication and integration services"),
        UNKNOWN("UNKNOWN", "Component type not classified");
        
        private final String code;
        private final String description;
        
        LraComponentType(String code, String description) {
            this.code = code;
            this.description = description;
        }
        
        public String getCode() {
            return code;
        }
        
        public String getDescription() {
            return description;
        }
        
        /**
         * Get an LraComponentType from its code
         * 
         * @param code The code to look up
         * @return The matching LraComponentType or UNKNOWN if not found
         */
        public static LraComponentType fromCode(String code) {
            if (code == null) {
                return UNKNOWN;
            }
            
            for (LraComponentType type : values()) {
                if (type.code.equalsIgnoreCase(code)) {
                    return type;
                }
            }
            return UNKNOWN;
        }
    }
    
    /**
     * Class representing a workload profile with detailed resource requirements
     */
    public static class WorkloadProfile {
        private String workloadId;
        private String workloadName;
        private BatchJobType batchType;
        private LraComponentType componentType;
        private boolean isLra;
        
        // Resource sensitivities (0-10 scale where 10 is highest sensitivity)
        private int cpuSensitivity;
        private int memorySensitivity;
        private int llcSensitivity;
        private int mbwSensitivity;
        private int ioSensitivity;
        
        // Mean resource utilization
        private double meanCpuUtil;
        private double meanMemoryUtil;
        private double meanLlcUtil;
        private double meanMbwUtil;
        private double meanIoUtil;
        
        public WorkloadProfile(String workloadId, String workloadName) {
            this.workloadId = workloadId;
            this.workloadName = workloadName;
            this.isLra = false;
            this.batchType = BatchJobType.UNKNOWN;
            this.componentType = LraComponentType.UNKNOWN;
        }
        
        /**
         * Create a classification profile based on observed metrics
         * 
         * @param metrics The collected metrics for this workload
         * @return The updated workload profile
         */
        public WorkloadProfile updateFromMetrics(Metrics metrics) {
            // Update mean utilization with exponential moving average (alpha = 0.2)
            double alpha = 0.2;
            this.meanCpuUtil = alpha * metrics.getCpuUtilization() + (1 - alpha) * this.meanCpuUtil;
            this.meanMemoryUtil = alpha * metrics.getMemoryUtilization() + (1 - alpha) * this.meanMemoryUtil;
            this.meanLlcUtil = alpha * metrics.getLlcUtilization() + (1 - alpha) * this.meanLlcUtil;
            this.meanMbwUtil = alpha * metrics.getMemoryBandwidth() + (1 - alpha) * this.meanMbwUtil;
            this.meanIoUtil = alpha * metrics.getIoThroughput() + (1 - alpha) * this.meanIoUtil;
            
            // Update classification if needed
            if (batchType == BatchJobType.UNKNOWN && !isLra) {
                classifyBatchType();
            }
            
            return this;
        }
        
        /**
         * Classify batch job type based on observed resource utilization
         */
        private void classifyBatchType() {
            // Simple classification based on mean utilization patterns
            
            // CPU-intensive if CPU utilization is highest
            if (meanCpuUtil > 70 && meanIoUtil < 40) {
                batchType = BatchJobType.CPU_INTENSIVE;
                cpuSensitivity = 9;
                ioSensitivity = 3;
            } 
            // IO-intensive if IO throughput is highest
            else if (meanIoUtil > 70 && meanCpuUtil < 40) {
                batchType = BatchJobType.IO_INTENSIVE;
                cpuSensitivity = 3;
                ioSensitivity = 9;
            }
            // Memory-intensive if memory utilization is highest
            else if (meanMemoryUtil > 70 && meanCpuUtil < 60 && meanIoUtil < 60) {
                batchType = BatchJobType.MEMORY_INTENSIVE;
                memorySensitivity = 9;
                cpuSensitivity = 5;
                ioSensitivity = 4;
            }
            // Hybrid if both CPU and IO are significant
            else if (meanCpuUtil > 50 && meanIoUtil > 50) {
                batchType = BatchJobType.HYBRID;
                cpuSensitivity = 7;
                ioSensitivity = 7;
            }
            // Otherwise remain unknown until more data collected
        }
        
        // Getters and setters
        
        public String getWorkloadId() {
            return workloadId;
        }
        
        public void setWorkloadId(String workloadId) {
            this.workloadId = workloadId;
        }
        
        public String getWorkloadName() {
            return workloadName;
        }
        
        public void setWorkloadName(String workloadName) {
            this.workloadName = workloadName;
        }
        
        public BatchJobType getBatchType() {
            return batchType;
        }
        
        public void setBatchType(BatchJobType batchType) {
            this.batchType = batchType;
        }
        
        public LraComponentType getComponentType() {
            return componentType;
        }
        
        public void setComponentType(LraComponentType componentType) {
            this.componentType = componentType;
        }
        
        public boolean isLra() {
            return isLra;
        }
        
        public void setLra(boolean lra) {
            isLra = lra;
        }
        
        public int getCpuSensitivity() {
            return cpuSensitivity;
        }
        
        public void setCpuSensitivity(int cpuSensitivity) {
            this.cpuSensitivity = cpuSensitivity;
        }
        
        public int getMemorySensitivity() {
            return memorySensitivity;
        }
        
        public void setMemorySensitivity(int memorySensitivity) {
            this.memorySensitivity = memorySensitivity;
        }
        
        public int getLlcSensitivity() {
            return llcSensitivity;
        }
        
        public void setLlcSensitivity(int llcSensitivity) {
            this.llcSensitivity = llcSensitivity;
        }
        
        public int getMbwSensitivity() {
            return mbwSensitivity;
        }
        
        public void setMbwSensitivity(int mbwSensitivity) {
            this.mbwSensitivity = mbwSensitivity;
        }
        
        public int getIoSensitivity() {
            return ioSensitivity;
        }
        
        public void setIoSensitivity(int ioSensitivity) {
            this.ioSensitivity = ioSensitivity;
        }
        
        public double getMeanCpuUtil() {
            return meanCpuUtil;
        }
        
        public void setMeanCpuUtil(double meanCpuUtil) {
            this.meanCpuUtil = meanCpuUtil;
        }
        
        public double getMeanMemoryUtil() {
            return meanMemoryUtil;
        }
        
        public void setMeanMemoryUtil(double meanMemoryUtil) {
            this.meanMemoryUtil = meanMemoryUtil;
        }
        
        public double getMeanLlcUtil() {
            return meanLlcUtil;
        }
        
        public void setMeanLlcUtil(double meanLlcUtil) {
            this.meanLlcUtil = meanLlcUtil;
        }
        
        public double getMeanMbwUtil() {
            return meanMbwUtil;
        }
        
        public void setMeanMbwUtil(double meanMbwUtil) {
            this.meanMbwUtil = meanMbwUtil;
        }
        
        public double getMeanIoUtil() {
            return meanIoUtil;
        }
        
        public void setMeanIoUtil(double meanIoUtil) {
            this.meanIoUtil = meanIoUtil;
        }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("WorkloadProfile[");
            sb.append("id=").append(workloadId);
            sb.append(", name=").append(workloadName);
            sb.append(", isLra=").append(isLra);
            
            if (isLra) {
                sb.append(", componentType=").append(componentType);
            } else {
                sb.append(", batchType=").append(batchType);
            }
            
            sb.append(", sensitivities=[cpu=").append(cpuSensitivity);
            sb.append(", mem=").append(memorySensitivity);
            sb.append(", llc=").append(llcSensitivity);
            sb.append(", mbw=").append(mbwSensitivity);
            sb.append(", io=").append(ioSensitivity).append("]");
            
            sb.append(", utilization=[cpu=").append(String.format("%.2f", meanCpuUtil));
            sb.append(", mem=").append(String.format("%.2f", meanMemoryUtil));
            sb.append(", llc=").append(String.format("%.2f", meanLlcUtil));
            sb.append(", mbw=").append(String.format("%.2f", meanMbwUtil));
            sb.append(", io=").append(String.format("%.2f", meanIoUtil)).append("]");
            
            sb.append("]");
            return sb.toString();
        }
    }
    
    /**
     * Helper method to create a workload profile for an LRA component
     * 
     * @param componentId Component identifier
     * @param componentName Component name
     * @param componentType Type of the LRA component
     * @return Initialized workload profile
     */
    public static WorkloadProfile createLraProfile(String componentId, String componentName, 
                                                 LraComponentType componentType) {
        WorkloadProfile profile = new WorkloadProfile(componentId, componentName);
        profile.setLra(true);
        profile.setComponentType(componentType);
        
        // Set default sensitivities based on component type
        switch (componentType) {
            case FRONTEND:
                profile.setCpuSensitivity(7);
                profile.setMemorySensitivity(5);
                profile.setLlcSensitivity(8);
                profile.setMbwSensitivity(4);
                profile.setIoSensitivity(2);
                break;
            case BACKEND:
                profile.setCpuSensitivity(9);
                profile.setMemorySensitivity(6);
                profile.setLlcSensitivity(7);
                profile.setMbwSensitivity(5);
                profile.setIoSensitivity(4);
                break;
            case DATABASE:
                profile.setCpuSensitivity(6);
                profile.setMemorySensitivity(8);
                profile.setLlcSensitivity(6);
                profile.setMbwSensitivity(7);
                profile.setIoSensitivity(9);
                break;
            case MIDDLEWARE:
                profile.setCpuSensitivity(5);
                profile.setMemorySensitivity(4);
                profile.setLlcSensitivity(5);
                profile.setMbwSensitivity(6);
                profile.setIoSensitivity(3);
                break;
            default:
                // Default sensitivities for unknown component types
                profile.setCpuSensitivity(5);
                profile.setMemorySensitivity(5);
                profile.setLlcSensitivity(5);
                profile.setMbwSensitivity(5);
                profile.setIoSensitivity(5);
        }
        
        return profile;
    }
    
    /**
     * Helper method to create a workload profile for a batch job
     * 
     * @param jobId Job identifier
     * @param jobName Job name
     * @param batchType Type of the batch job
     * @return Initialized workload profile
     */
    public static WorkloadProfile createBatchProfile(String jobId, String jobName, 
                                                   BatchJobType batchType) {
        WorkloadProfile profile = new WorkloadProfile(jobId, jobName);
        profile.setLra(false);
        profile.setBatchType(batchType);
        
        // Set default sensitivities based on batch type
        switch (batchType) {
            case CPU_INTENSIVE:
                profile.setCpuSensitivity(9);
                profile.setMemorySensitivity(5);
                profile.setLlcSensitivity(7);
                profile.setMbwSensitivity(4);
                profile.setIoSensitivity(2);
                break;
            case IO_INTENSIVE:
                profile.setCpuSensitivity(3);
                profile.setMemorySensitivity(4);
                profile.setLlcSensitivity(3);
                profile.setMbwSensitivity(5);
                profile.setIoSensitivity(9);
                break;
            case MEMORY_INTENSIVE:
                profile.setCpuSensitivity(4);
                profile.setMemorySensitivity(9);
                profile.setLlcSensitivity(6);
                profile.setMbwSensitivity(8);
                profile.setIoSensitivity(3);
                break;
            case HYBRID:
                profile.setCpuSensitivity(7);
                profile.setMemorySensitivity(6);
                profile.setLlcSensitivity(5);
                profile.setMbwSensitivity(6);
                profile.setIoSensitivity(7);
                break;
            default:
                // Default sensitivities for unknown batch types
                profile.setCpuSensitivity(5);
                profile.setMemorySensitivity(5);
                profile.setLlcSensitivity(5);
                profile.setMbwSensitivity(5);
                profile.setIoSensitivity(5);
        }
        
        return profile;
    }
}