package com.perphproctor.appcontroller;

import com.perphproctor.common.Constants;
import com.perphproctor.common.Metrics;
import com.perphproctor.common.Utils;

import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.api.records.NodeId;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * MetricsCollector is responsible for gathering multi-dimensional resource metrics
 * from both system level and container level. It leverages various monitoring tools
 * to collect CPU, memory, last-level cache (LLC), memory bandwidth (MBW), and I/O
 * throughput data.
 * 
 * This class provides the foundation for the monitoring capabilities of the
 * AppController component in the PerphProctor framework.
 */
public class MetricsCollector implements Serializable {
    
    private static final long serialVersionUID = 1L;
    private static final Logger LOG = LoggerFactory.getLogger(MetricsCollector.class);
    
    // Node identification
    private final NodeId nodeId;
    private final String hostname;
    
    // Collection state
    private final AtomicBoolean running;
    
    // Command paths
    private final String perfToolPath;
    private final String dockerStatsCmdPath;
    private final String intelRdtToolPath;
    
    // Metric caches
    private final Map<ContainerId, Metrics> containerMetricsCache;
    private Metrics nodeMetricsCache;
    private long lastNodeMetricsUpdateTime;
    
    // Cache validity period (ms)
    private static final long METRICS_CACHE_VALIDITY_PERIOD = 1000; // 1 second
    
    // Metrics collection commands and patterns
    private static final String CPU_STAT_FILE = "/proc/stat";
    private static final String MEMORY_STAT_FILE = "/proc/meminfo";
    private static final String IO_STAT_FILE = "/proc/diskstats";
    private static final String DOCKER_STATS_CMD = "docker stats --no-stream --format \"{{.ID}}|{{.CPUPerc}}|{{.MemPerc}}|{{.MemUsage}}|{{.NetIO}}|{{.BlockIO}}\"";
    private static final String INTEL_RDT_LLC_CMD = "pqos -r --show-LLC";
    private static final String INTEL_RDT_MBW_CMD = "pqos -r --show-MBW-local";
    
    /**
     * Constructor with default tool paths
     * 
     * @param nodeId Node identifier
     */
    public MetricsCollector(NodeId nodeId) {
        this(
            nodeId,
            "/usr/bin/perf",
            "/usr/bin/docker",
            "/usr/bin/pqos"
        );
    }
    
    /**
     * Constructor with custom tool paths
     * 
     * @param nodeId Node identifier
     * @param perfToolPath Path to perf tool
     * @param dockerStatsCmdPath Path to docker command
     * @param intelRdtToolPath Path to Intel RDT tool
     */
    public MetricsCollector(NodeId nodeId, String perfToolPath, 
                         String dockerStatsCmdPath, String intelRdtToolPath) {
        this.nodeId = nodeId;
        this.hostname = Utils.getHostname();
        this.perfToolPath = perfToolPath;
        this.dockerStatsCmdPath = dockerStatsCmdPath;
        this.intelRdtToolPath = intelRdtToolPath;
        
        this.running = new AtomicBoolean(false);
        this.containerMetricsCache = new ConcurrentHashMap<>();
        this.nodeMetricsCache = new Metrics();
        this.lastNodeMetricsUpdateTime = 0;
        
        LOG.info("MetricsCollector initialized for node {} (hostname: {})", 
                nodeId, hostname);
    }
    
    /**
     * Start metrics collection
     * 
     * @return true if started successfully
     */
    public boolean start() {
        if (running.compareAndSet(false, true)) {
            LOG.info("Starting metrics collection for node {}", nodeId);
            
            // Verify tool availability
            boolean toolsAvailable = verifyToolAvailability();
            if (!toolsAvailable) {
                LOG.warn("Some monitoring tools are not available, metrics collection may be limited");
            }
            
            // Initialize metrics cache
            updateNodeMetrics();
            
            return true;
        } else {
            LOG.warn("Metrics collection already running for node {}", nodeId);
            return false;
        }
    }
    
    /**
     * Stop metrics collection
     * 
     * @return true if stopped successfully
     */
    public boolean stop() {
        if (running.compareAndSet(true, false)) {
            LOG.info("Stopping metrics collection for node {}", nodeId);
            return true;
        } else {
            LOG.warn("Metrics collection already stopped for node {}", nodeId);
            return false;
        }
    }
    
    /**
     * Collect node-level metrics
     * 
     * @return Node metrics
     */
    public Metrics collectNodeMetrics() {
        if (!running.get()) {
            LOG.warn("Metrics collection is not running");
            return new Metrics();
        }
        
        // Check if cached metrics are valid
        long currentTime = System.currentTimeMillis();
        if (currentTime - lastNodeMetricsUpdateTime < METRICS_CACHE_VALIDITY_PERIOD) {
            return nodeMetricsCache;
        }
        
        // Update metrics cache
        updateNodeMetrics();
        
        return nodeMetricsCache;
    }
    
    /**
     * Collect container-level metrics
     * 
     * @param containerIds List of container IDs to collect metrics for
     * @return Map of container IDs to metrics
     */
    public Map<ContainerId, Metrics> collectContainerMetrics(List<ContainerId> containerIds) {
        if (!running.get()) {
            LOG.warn("Metrics collection is not running");
            return new HashMap<>();
        }
        
        if (containerIds == null || containerIds.isEmpty()) {
            return new HashMap<>();
        }
        
        Map<ContainerId, Metrics> containerMetrics = new HashMap<>();
        
        // Collect Docker container metrics
        Map<String, Metrics> dockerMetrics = collectDockerMetrics();
        
        // Map Docker container IDs to YARN container IDs
        for (ContainerId containerId : containerIds) {
            String dockerId = getDockerIdForContainer(containerId);
            if (dockerId != null && dockerMetrics.containsKey(dockerId)) {
                containerMetrics.put(containerId, dockerMetrics.get(dockerId));
            } else {
                // If Docker metrics not available, create empty metrics
                containerMetrics.put(containerId, new Metrics());
            }
        }
        
        // Update cache
        containerMetricsCache.putAll(containerMetrics);
        
        return containerMetrics;
    }
    
    /**
     * Get cached metrics for a container
     * 
     * @param containerId Container ID
     * @return Container metrics or null if not cached
     */
    public Metrics getCachedContainerMetrics(ContainerId containerId) {
        if (containerId == null) {
            return null;
        }
        
        return containerMetricsCache.get(containerId);
    }
    
    /**
     * Update node metrics cache
     */
    private void updateNodeMetrics() {
        Metrics metrics = new Metrics();
        
        try {
            // Collect CPU metrics
            metrics.setCpuUtilization(collectCpuUtilization());
            
            // Collect memory metrics
            metrics.setMemoryUtilization(collectMemoryUtilization());
            
            // Collect LLC metrics
            metrics.setLlcUtilization(collectLlcUtilization());
            
            // Collect MBW metrics
            metrics.setMemoryBandwidth(collectMemoryBandwidth());
            
            // Collect I/O metrics
            metrics.setIoThroughput(collectIoThroughput());
            
            // Update cache
            nodeMetricsCache = metrics;
            lastNodeMetricsUpdateTime = System.currentTimeMillis();
            
            LOG.debug("Updated node metrics cache: {}", metrics);
        } catch (Exception e) {
            LOG.error("Error updating node metrics: {}", e.getMessage(), e);
        }
    }
    
    /**
     * Collect CPU utilization percentage
     * 
     * @return CPU utilization percentage (0-100)
     */
    private double collectCpuUtilization() {
        try {
            // Read /proc/stat for CPU metrics
            List<String> lines = Files.readAllLines(Paths.get(CPU_STAT_FILE));
            
            // Parse CPU line
            // Format: cpu user nice system idle iowait irq softirq steal guest guest_nice
            if (!lines.isEmpty()) {
                String cpuLine = lines.get(0);
                if (cpuLine.startsWith("cpu ")) {
                    String[] parts = cpuLine.split("\\s+");
                    if (parts.length >= 8) {
                        long user = Long.parseLong(parts[1]);
                        long nice = Long.parseLong(parts[2]);
                        long system = Long.parseLong(parts[3]);
                        long idle = Long.parseLong(parts[4]);
                        long iowait = Long.parseLong(parts[5]);
                        long irq = Long.parseLong(parts[6]);
                        long softirq = Long.parseLong(parts[7]);
                        
                        long totalCpuTime = user + nice + system + idle + iowait + irq + softirq;
                        long idleTime = idle + iowait;
                        long activeTime = totalCpuTime - idleTime;
                        
                        return (activeTime * 100.0) / totalCpuTime;
                    }
                }
            }
            
            // Fallback to system API
            return Utils.getSystemCpuUtilization();
        } catch (Exception e) {
            LOG.error("Error collecting CPU utilization: {}", e.getMessage());
            return Utils.getSystemCpuUtilization();
        }
    }
    
    /**
     * Collect memory utilization percentage
     * 
     * @return Memory utilization percentage (0-100)
     */
    private double collectMemoryUtilization() {
        try {
            // Read /proc/meminfo for memory metrics
            List<String> lines = Files.readAllLines(Paths.get(MEMORY_STAT_FILE));
            
            long totalMemory = 0;
            long freeMemory = 0;
            long buffersMemory = 0;
            long cachedMemory = 0;
            
            for (String line : lines) {
                if (line.startsWith("MemTotal:")) {
                    totalMemory = parseMemValue(line);
                } else if (line.startsWith("MemFree:")) {
                    freeMemory = parseMemValue(line);
                } else if (line.startsWith("Buffers:")) {
                    buffersMemory = parseMemValue(line);
                } else if (line.startsWith("Cached:")) {
                    cachedMemory = parseMemValue(line);
                }
            }
            
            if (totalMemory > 0) {
                // Calculate used memory (excluding buffers/cache)
                long usedMemory = totalMemory - freeMemory - buffersMemory - cachedMemory;
                return (usedMemory * 100.0) / totalMemory;
            }
            
            // Fallback to JVM memory as approximation
            return Utils.getJvmMemoryUtilization();
        } catch (Exception e) {
            LOG.error("Error collecting memory utilization: {}", e.getMessage());
            return Utils.getJvmMemoryUtilization();
        }
    }
    
    /**
     * Parse memory value from /proc/meminfo
     * 
     * @param line Line from /proc/meminfo
     * @return Memory value in KB
     */
    private long parseMemValue(String line) {
        String[] parts = line.split("\\s+");
        if (parts.length >= 2) {
            try {
                return Long.parseLong(parts[1]);
            } catch (NumberFormatException e) {
                return 0;
            }
        }
        return 0;
    }
    
    /**
     * Collect last-level cache utilization percentage
     * 
     * @return LLC utilization percentage (0-100)
     */
    private double collectLlcUtilization() {
        if (!isIntelRdtAvailable()) {
            // LLC monitoring not available, return default value
            return 0.0;
        }
        
        try {
            // Use Intel RDT tool to get LLC metrics
            String output = Utils.executeCommand(INTEL_RDT_LLC_CMD);
            
            // Parse output to extract LLC usage
            // Example output: "Core 0-3; LLC: 5.5MB"
            Pattern pattern = Pattern.compile("LLC:\\s+(\\d+\\.?\\d*)MB");
            Matcher matcher = pattern.matcher(output);
            
            if (matcher.find()) {
                String llcUsageStr = matcher.group(1);
                double llcUsageMB = Double.parseDouble(llcUsageStr);
                
                // Get total LLC size (assumed to be 11MB from Constants)
                double totalLlcMB = 11.0; // From system configuration
                
                return (llcUsageMB * 100.0) / totalLlcMB;
            }
        } catch (Exception e) {
            LOG.error("Error collecting LLC utilization: {}", e.getMessage());
        }
        
        return 0.0;
    }
    
    /**
     * Collect memory bandwidth usage
     * 
     * @return Memory bandwidth in MB/s
     */
    private double collectMemoryBandwidth() {
        if (!isIntelRdtAvailable()) {
            // MBW monitoring not available, return default value
            return 0.0;
        }
        
        try {
            // Use Intel RDT tool to get MBW metrics
            String output = Utils.executeCommand(INTEL_RDT_MBW_CMD);
            
            // Parse output to extract MBW usage
            // Example output: "MBW local: 1250.5MB/s"
            Pattern pattern = Pattern.compile("MBW local:\\s+(\\d+\\.?\\d*)MB\\/s");
            Matcher matcher = pattern.matcher(output);
            
            if (matcher.find()) {
                String mbwUsageStr = matcher.group(1);
                return Double.parseDouble(mbwUsageStr);
            }
        } catch (Exception e) {
            LOG.error("Error collecting memory bandwidth: {}", e.getMessage());
        }
        
        return 0.0;
    }
    
    /**
     * Collect I/O throughput
     * 
     * @return I/O throughput in IOPS
     */
    private double collectIoThroughput() {
        try {
            // Read /proc/diskstats for I/O metrics
            List<String> lines = Files.readAllLines(Paths.get(IO_STAT_FILE));
            
            double totalIops = 0.0;
            
            for (String line : lines) {
                String[] parts = line.trim().split("\\s+");
                if (parts.length >= 14) {
                    // Fields 4 and 8 are reads and writes completed
                    int reads = Integer.parseInt(parts[3]);
                    int writes = Integer.parseInt(parts[7]);
                    
                    totalIops += reads + writes;
                }
            }
            
            return totalIops;
        } catch (Exception e) {
            LOG.error("Error collecting I/O throughput: {}", e.getMessage());
            return 0.0;
        }
    }
    
    /**
     * Collect Docker container metrics
     * 
     * @return Map of Docker container IDs to metrics
     */
    private Map<String, Metrics> collectDockerMetrics() {
        Map<String, Metrics> containerMetrics = new HashMap<>();
        
        if (!isDockerAvailable()) {
            LOG.debug("Docker is not available, skipping container metrics collection");
            return containerMetrics;
        }
        
        try {
            // Execute docker stats command
            String dockerStatsCmd = dockerStatsCmdPath + " stats --no-stream --format \"{{.ID}}|{{.CPUPerc}}|{{.MemPerc}}|{{.MemUsage}}|{{.NetIO}}|{{.BlockIO}}\"";
            String output = Utils.executeCommand(dockerStatsCmd);
            
            // Parse output lines
            String[] lines = output.split("\n");
            for (String line : lines) {
                if (line.trim().isEmpty()) {
                    continue;
                }
                
                String[] parts = line.split("\\|");
                if (parts.length >= 6) {
                    String containerId = parts[0];
                    
                    Metrics metrics = new Metrics();
                    
                    // Parse CPU percentage
                    metrics.setCpuUtilization(parsePercentage(parts[1]));
                    
                    // Parse memory percentage
                    metrics.setMemoryUtilization(parsePercentage(parts[2]));
                    
                    // Parse Block I/O as an approximation for I/O throughput
                    metrics.setIoThroughput(parseBlockIO(parts[5]));
                    
                    containerMetrics.put(containerId, metrics);
                }
            }
        } catch (Exception e) {
            LOG.error("Error collecting Docker metrics: {}", e.getMessage());
        }
        
        return containerMetrics;
    }
    
    /**
     * Parse percentage string
     * 
     * @param percentStr Percentage string (e.g., "45.5%")
     * @return Percentage value (0-100)
     */
    private double parsePercentage(String percentStr) {
        if (percentStr == null || percentStr.trim().isEmpty()) {
            return 0.0;
        }
        
        try {
            // Remove % symbol and parse
            String valueStr = percentStr.trim().replace("%", "");
            return Double.parseDouble(valueStr);
        } catch (NumberFormatException e) {
            return 0.0;
        }
    }
    
    /**
     * Parse Block I/O string
     * 
     * @param blockIOStr Block I/O string (e.g., "10.2MB / 20.5MB")
     * @return I/O throughput approximation
     */
    private double parseBlockIO(String blockIOStr) {
        if (blockIOStr == null || blockIOStr.trim().isEmpty()) {
            return 0.0;
        }
        
        try {
            // Extract total I/O (reads + writes)
            String[] parts = blockIOStr.split("/");
            if (parts.length >= 2) {
                // Extract first number as approximation
                String readStr = parts[0].trim().replaceAll("[^0-9.]", "");
                String writeStr = parts[1].trim().replaceAll("[^0-9.]", "");
                
                double readValue = Double.parseDouble(readStr);
                double writeValue = Double.parseDouble(writeStr);
                
                return readValue + writeValue;
            }
        } catch (Exception e) {
            // Ignore parsing errors
        }
        
        return 0.0;
    }
    
    /**
     * Get Docker container ID for YARN container
     * 
     * @param containerId YARN container ID
     * @return Docker container ID or null if not found
     */
    private String getDockerIdForContainer(ContainerId containerId) {
        if (containerId == null) {
            return null;
        }
        
        try {
            // Use container ID as search term in docker ps
            String dockerPsCmd = dockerStatsCmdPath + " ps --format \"{{.ID}}\" --filter label=org.apache.hadoop.yarn.server.nodemanager.containermanager.container.id=" + containerId.toString();
            String output = Utils.executeCommand(dockerPsCmd);
            
            // Return first line as Docker ID
            String[] lines = output.split("\n");
            if (lines.length > 0 && !lines[0].trim().isEmpty()) {
                return lines[0].trim();
            }
        } catch (Exception e) {
            LOG.error("Error mapping YARN container to Docker container: {}", e.getMessage());
        }
        
        return null;
    }
    
    /**
     * Verify availability of monitoring tools
     * 
     * @return true if all tools are available
     */
    private boolean verifyToolAvailability() {
        boolean allAvailable = true;
        
        // Check perf tool
        if (!isPerfAvailable()) {
            LOG.warn("Perf tool not available at {}", perfToolPath);
            allAvailable = false;
        }
        
        // Check Docker
        if (!isDockerAvailable()) {
            LOG.warn("Docker not available at {}", dockerStatsCmdPath);
            allAvailable = false;
        }
        
        // Check Intel RDT tools
        if (!isIntelRdtAvailable()) {
            LOG.warn("Intel RDT tool not available at {}", intelRdtToolPath);
            allAvailable = false;
        }
        
        return allAvailable;
    }
    
    /**
     * Check if perf tool is available
     * 
     * @return true if available
     */
    private boolean isPerfAvailable() {
        return isToolAvailable(perfToolPath);
    }
    
    /**
     * Check if Docker is available
     * 
     * @return true if available
     */
    private boolean isDockerAvailable() {
        return isToolAvailable(dockerStatsCmdPath);
    }
    
    /**
     * Check if Intel RDT tool is available
     * 
     * @return true if available
     */
    private boolean isIntelRdtAvailable() {
        return isToolAvailable(intelRdtToolPath);
    }
    
    /**
     * Check if a tool is available by checking file existence and execution permission
     * 
     * @param toolPath Path to tool
     * @return true if tool exists and is executable
     */
    private boolean isToolAvailable(String toolPath) {
        if (toolPath == null || toolPath.isEmpty()) {
            return false;
        }
        
        File toolFile = new File(toolPath);
        return toolFile.exists() && toolFile.canExecute();
    }
}