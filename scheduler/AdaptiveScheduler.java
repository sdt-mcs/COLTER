package com.perphproctor.scheduler;

import com.perphproctor.common.Constants;
import com.perphproctor.common.Metrics;
import com.perphproctor.common.Utils;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.api.records.Container;
import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.api.records.NodeId;
import org.apache.hadoop.yarn.api.records.Priority;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.server.resourcemanager.rmcontainer.RMContainer;
import org.apache.hadoop.yarn.server.resourcemanager.rmcontainer.RMContainerState;
import org.apache.hadoop.yarn.server.resourcemanager.scheduler.AbstractYarnScheduler;
import org.apache.hadoop.yarn.server.resourcemanager.scheduler.SchedulerNode;
import org.apache.hadoop.yarn.server.resourcemanager.scheduler.capacity.CapacityScheduler;
import org.apache.hadoop.yarn.server.resourcemanager.scheduler.common.fica.FiCaSchedulerApp;
import org.apache.hadoop.yarn.server.resourcemanager.scheduler.common.fica.FiCaSchedulerNode;
import org.apache.hadoop.conf.Configuration;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * AdaptiveScheduler implements QoS-aware co-scheduling algorithm based on 
 * continual learning performance prediction. It extends YARN's capacity
 * scheduler with additional logic to ensure LRA performance guarantees.
 * 
 * The scheduler incorporates node recommendation scores to make informed
 * placement decisions, preventing performance interference between co-located
 * workloads.
 */
public class AdaptiveScheduler {
    
    private static final Logger LOG = LoggerFactory.getLogger(AdaptiveScheduler.class);
    
    // Configuration parameters
    private final double sensitivityParameter;
    private final int maxTaskRetryCount;
    
    // Underlying YARN scheduler
    private final AbstractYarnScheduler scheduler;
    
    // Node recommendation scores
    private final Map<NodeId, Double> nodeRecommendationScores;
    
    // Task retry counters
    private final Map<ContainerId, Integer> taskRetryCounters;
    
    // Cache of node scheduling decisions
    private final Map<NodeId, Boolean> nodeAcceptanceCache;
    private long lastCacheClearTime;
    
    // Task data locality tracking
    private final Map<ContainerId, List<NodeId>> taskLocalityInfo;
    
    // Thread safety
    private final ReentrantReadWriteLock lock;
    
    // Scheduler components
    private final TaskPlacementManager taskPlacementManager;
    private final QoSAwareCoScheduler qoSAwareCoScheduler;
    private final ThresholdManager thresholdManager;
    
    /**
     * Constructor for AdaptiveScheduler
     * 
     * @param scheduler Underlying YARN scheduler
     * @param config Configuration parameters
     */
    public AdaptiveScheduler(AbstractYarnScheduler scheduler, Configuration config) {
        this.scheduler = scheduler;
        
        // Initialize configuration parameters
        this.sensitivityParameter = config.getDouble(
                "perphproctor.scheduler.sensitivity.parameter",
                Constants.DEFAULT_SENSITIVITY_PARAMETER_K);
        
        this.maxTaskRetryCount = config.getInt(
                "perphproctor.scheduler.max.task.retry.count",
                Constants.MAX_TASK_RETRY_COUNT);
        
        // Initialize data structures
        this.nodeRecommendationScores = new ConcurrentHashMap<>();
        this.taskRetryCounters = new ConcurrentHashMap<>();
        this.nodeAcceptanceCache = new ConcurrentHashMap<>();
        this.taskLocalityInfo = new ConcurrentHashMap<>();
        this.lastCacheClearTime = System.currentTimeMillis();
        
        // Initialize thread safety
        this.lock = new ReentrantReadWriteLock();
        
        // Initialize scheduler components
        this.taskPlacementManager = new TaskPlacementManager(this);
        this.qoSAwareCoScheduler = new QoSAwareCoScheduler(this);
        this.thresholdManager = new ThresholdManager(this);
        
        LOG.info("AdaptiveScheduler initialized with sensitivity parameter k={}", sensitivityParameter);
    }
    
    /**
     * Process node recommendation scores from the AppController
     * 
     * @param nodeId Node identifier
     * @param score Recommendation score (0-1)
     */
    public void updateNodeRecommendation(NodeId nodeId, double score) {
        if (nodeId == null) {
            return;
        }
        
        lock.writeLock().lock();
        try {
            nodeRecommendationScores.put(nodeId, score);
            
            // Clear cache periodically
            long currentTime = System.currentTimeMillis();
            if (currentTime - lastCacheClearTime > 60000) { // Clear every minute
                nodeAcceptanceCache.clear();
                lastCacheClearTime = currentTime;
            }
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Check if a node can accept a new task based on recommendation score
     * 
     * @param nodeId Node identifier
     * @return true if node can accept task, false otherwise
     */
    public boolean canNodeAcceptTask(NodeId nodeId) {
        if (nodeId == null) {
            return false;
        }
        
        // Check cache first
        Boolean cachedResult = nodeAcceptanceCache.get(nodeId);
        if (cachedResult != null) {
            return cachedResult;
        }
        
        lock.readLock().lock();
        try {
            // Get recommendation score for this node
            Double score = nodeRecommendationScores.getOrDefault(nodeId, 0.5);
            
            // Calculate threshold based on current cluster state
            double threshold = thresholdManager.calculateAcceptanceThreshold();
            
            // Make decision
            boolean canAccept = score >= threshold;
            
            // Cache result
            nodeAcceptanceCache.put(nodeId, canAccept);
            
            return canAccept;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Process task retry information
     * 
     * @param containerId Container identifier
     * @return Current retry count
     */
    public int incrementTaskRetryCount(ContainerId containerId) {
        if (containerId == null) {
            return 0;
        }
        
        lock.writeLock().lock();
        try {
            int count = taskRetryCounters.getOrDefault(containerId, 0);
            count++;
            taskRetryCounters.put(containerId, count);
            return count;
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Get current retry count for a task
     * 
     * @param containerId Container identifier
     * @return Current retry count
     */
    public int getTaskRetryCount(ContainerId containerId) {
        if (containerId == null) {
            return 0;
        }
        
        lock.readLock().lock();
        try {
            return taskRetryCounters.getOrDefault(containerId, 0);
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Register data locality information for a task
     * 
     * @param containerId Container identifier
     * @param localNodes List of nodes with data locality
     */
    public void registerTaskLocalityInfo(ContainerId containerId, List<NodeId> localNodes) {
        if (containerId == null || localNodes == null) {
            return;
        }
        
        lock.writeLock().lock();
        try {
            taskLocalityInfo.put(containerId, new ArrayList<>(localNodes));
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Check if a node has data locality for a task
     * 
     * @param containerId Container identifier
     * @param nodeId Node identifier
     * @return true if node has data locality for task
     */
    public boolean hasDataLocality(ContainerId containerId, NodeId nodeId) {
        if (containerId == null || nodeId == null) {
            return false;
        }
        
        lock.readLock().lock();
        try {
            List<NodeId> localNodes = taskLocalityInfo.get(containerId);
            if (localNodes == null) {
                return false;
            }
            
            return localNodes.contains(nodeId);
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Get all nodes with data locality for a task
     * 
     * @param containerId Container identifier
     * @return List of nodes with data locality
     */
    public List<NodeId> getLocalityNodes(ContainerId containerId) {
        if (containerId == null) {
            return new ArrayList<>();
        }
        
        lock.readLock().lock();
        try {
            List<NodeId> localNodes = taskLocalityInfo.get(containerId);
            if (localNodes == null) {
                return new ArrayList<>();
            }
            
            return new ArrayList<>(localNodes);
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Check if a task should be scheduled despite node rejection
     * 
     * @param containerId Container identifier
     * @param nodeId Node identifier
     * @return true if task should be scheduled anyway
     */
    public boolean shouldScheduleAnyway(ContainerId containerId, NodeId nodeId) {
        // Check retry count
        int retryCount = getTaskRetryCount(containerId);
        if (retryCount >= maxTaskRetryCount) {
            
            // If exceeded max retries, try to schedule
            // Priority for data-local placement
            if (hasDataLocality(containerId, nodeId)) {
                return true;
            }
            
            // For non-local placement, use random chance based on recommendation score
            double score = nodeRecommendationScores.getOrDefault(nodeId, 0.5);
            double randomValue = Math.random();
            
            // Higher score = higher chance of acceptance
            return randomValue < score;
        }
        
        return false;
    }
    
    /**
     * Check if a container should be scheduled on a node
     * 
     * @param container Container to schedule
     * @param node Target node
     * @return true if container should be scheduled on node
     */
    public boolean shouldScheduleContainer(RMContainer container, SchedulerNode node) {
        if (container == null || node == null) {
            return false;
        }
        
        ContainerId containerId = container.getContainerId();
        NodeId nodeId = node.getNodeID();
        
        // Check if node can accept task
        if (canNodeAcceptTask(nodeId)) {
            return true;
        }
        
        // If node rejected, check if we should schedule anyway
        return shouldScheduleAnyway(containerId, nodeId);
    }
    
    /**
     * Clear task tracking information
     * 
     * @param containerId Container identifier
     */
    public void clearTaskInfo(ContainerId containerId) {
        if (containerId == null) {
            return;
        }
        
        lock.writeLock().lock();
        try {
            taskRetryCounters.remove(containerId);
            taskLocalityInfo.remove(containerId);
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Get recommendation scores for all nodes
     * 
     * @return Map of node IDs to recommendation scores
     */
    public Map<NodeId, Double> getNodeRecommendationScores() {
        lock.readLock().lock();
        try {
            return new HashMap<>(nodeRecommendationScores);
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Get the underlying YARN scheduler
     * 
     * @return YARN scheduler
     */
    public AbstractYarnScheduler getScheduler() {
        return scheduler;
    }
    
    /**
     * Get the sensitivity parameter
     * 
     * @return Sensitivity parameter
     */
    public double getSensitivityParameter() {
        return sensitivityParameter;
    }
    
    /**
     * Get the task placement manager
     * 
     * @return Task placement manager
     */
    public TaskPlacementManager getTaskPlacementManager() {
        return taskPlacementManager;
    }
    
    /**
     * Get the QoS-aware co-scheduler
     * 
     * @return QoS-aware co-scheduler
     */
    public QoSAwareCoScheduler getQoSAwareCoScheduler() {
        return qoSAwareCoScheduler;
    }
    
    /**
     * Get the threshold manager
     * 
     * @return Threshold manager
     */
    public ThresholdManager getThresholdManager() {
        return thresholdManager;
    }
}