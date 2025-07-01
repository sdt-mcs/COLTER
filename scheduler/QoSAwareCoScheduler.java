package com.perphproctor.scheduler;

import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.api.records.NodeId;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.server.resourcemanager.rmcontainer.RMContainer;
import org.apache.hadoop.yarn.server.resourcemanager.scheduler.SchedulerNode;
import org.apache.hadoop.yarn.server.resourcemanager.scheduler.common.fica.FiCaSchedulerApp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.LinkedList;

/**
 * QoSAwareCoScheduler implements Algorithm 3 from the paper: 
 * "QoS-aware scheduling based on co-location forecasting."
 * 
 * It manages the co-scheduling of batch jobs with long-running applications
 * (LRAs) to ensure QoS guarantees while maximizing resource utilization.
 */
public class QoSAwareCoScheduler {
    
    private static final Logger LOG = LoggerFactory.getLogger(QoSAwareCoScheduler.class);
    
    // Reference to the main scheduler
    private final AdaptiveScheduler adaptiveScheduler;
    
    // Pending task queue
    private final Queue<SchedulingTask> pendingTasks;
    
    /**
     * Constructor
     * 
     * @param adaptiveScheduler Reference to the main adaptive scheduler
     */
    public QoSAwareCoScheduler(AdaptiveScheduler adaptiveScheduler) {
        this.adaptiveScheduler = adaptiveScheduler;
        this.pendingTasks = new LinkedList<>();
        
        LOG.info("QoSAwareCoScheduler initialized");
    }
    
    /**
     * Main scheduling method that implements Algorithm 3
     * 
     * @param pendingQueue Queue of tasks waiting to be scheduled
     * @param nodes Available nodes for scheduling
     * @return List of successful task placements
     */
    public List<TaskPlacement> scheduleQoSAware(List<SchedulingTask> pendingQueue, List<SchedulerNode> nodes) {
        List<TaskPlacement> placements = new ArrayList<>();
        
        // Sort tasks by waiting time in descending order
        List<SchedulingTask> sortedTasks = new ArrayList<>(pendingQueue);
        Collections.sort(sortedTasks, Comparator.comparing(SchedulingTask::getWaitingTime).reversed());
        
        // Try to schedule each task
        for (SchedulingTask task : sortedTasks) {
            boolean placed = false;
            
            // Try all nodes
            for (SchedulerNode node : nodes) {
                // Check if node has enough resources
                if (!hasEnoughResources(node, task.getResourceRequest())) {
                    continue;
                }
                
                // Check if node accepts the task
                if (node.getNodeID() != null && acceptTask(node, task)) {
                    // Place task on node
                    TaskPlacement placement = new TaskPlacement(task, node);
                    placements.add(placement);
                    placed = true;
                    break;
                }
            }
            
            // If task wasn't placed, increment retry counter
            if (!placed) {
                adaptiveScheduler.incrementTaskRetryCount(task.getContainerId());
                pendingTasks.add(task);
            }
        }
        
        return placements;
    }
    
    /**
     * Check if a node has enough resources for a task
     * 
     * @param node Node to check
     * @param resourceRequest Resource request
     * @return true if node has enough resources
     */
    private boolean hasEnoughResources(SchedulerNode node, Resource resourceRequest) {
        if (node == null || resourceRequest == null) {
            return false;
        }
        
        Resource available = node.getAvailableResource();
        
        // Check if available resources are sufficient
        return available.getMemorySize() >= resourceRequest.getMemorySize() &&
               available.getVirtualCores() >= resourceRequest.getVirtualCores();
    }
    
    /**
     * Determine if a task should be accepted by a node
     * 
     * @param node Node to check
     * @param task Task to check
     * @return true if node should accept task
     */
    private boolean acceptTask(SchedulerNode node, SchedulingTask task) {
        NodeId nodeId = node.getNodeID();
        ContainerId containerId = task.getContainerId();
        
        // Check if node can accept the task based on recommendation score
        if (adaptiveScheduler.canNodeAcceptTask(nodeId)) {
            return true;
        }
        
        // If locality exists and task has been retried at least once, accept
        if (adaptiveScheduler.hasDataLocality(containerId, nodeId) && 
            adaptiveScheduler.getTaskRetryCount(containerId) >= 1) {
            return true;
        }
        
        // For non-local tasks with retries, use probabilistic acceptance
        if (!adaptiveScheduler.hasDataLocality(containerId, nodeId) && 
            adaptiveScheduler.getTaskRetryCount(containerId) >= 1) {
            
            // Generate random score for comparison
            double randomScore = Math.random();
            
            // Get node's recommendation score
            double nodeScore = adaptiveScheduler.getNodeRecommendationScores()
                                              .getOrDefault(nodeId, 0.5);
            
            // Accept if random score is lower than node score
            return randomScore < nodeScore;
        }
        
        return false;
    }
    
    /**
     * Add a task to the pending queue
     * 
     * @param task Task to add
     */
    public void addPendingTask(SchedulingTask task) {
        if (task != null) {
            pendingTasks.add(task);
        }
    }
    
    /**
     * Get all pending tasks
     * 
     * @return List of pending tasks
     */
    public List<SchedulingTask> getPendingTasks() {
        return new ArrayList<>(pendingTasks);
    }
    
    /**
     * Clear pending tasks
     */
    public void clearPendingTasks() {
        pendingTasks.clear();
    }
    
    /**
     * Class representing a task placement decision
     */
    public static class TaskPlacement {
        private final SchedulingTask task;
        private final SchedulerNode node;
        
        public TaskPlacement(SchedulingTask task, SchedulerNode node) {
            this.task = task;
            this.node = node;
        }
        
        public SchedulingTask getTask() {
            return task;
        }
        
        public SchedulerNode getNode() {
            return node;
        }
        
        @Override
        public String toString() {
            return "TaskPlacement{" +
                   "task=" + task.getContainerId() +
                   ", node=" + node.getNodeID() +
                   '}';
        }
    }
    
    /**
     * Class representing a task to be scheduled
     */
    public static class SchedulingTask {
        private final ContainerId containerId;
        private final ApplicationId applicationId;
        private final Resource resourceRequest;
        private final long creationTime;
        private int retryCount;
        
        public SchedulingTask(ContainerId containerId, ApplicationId applicationId, 
                              Resource resourceRequest) {
            this.containerId = containerId;
            this.applicationId = applicationId;
            this.resourceRequest = resourceRequest;
            this.creationTime = System.currentTimeMillis();
            this.retryCount = 0;
        }
        
        public ContainerId getContainerId() {
            return containerId;
        }
        
        public ApplicationId getApplicationId() {
            return applicationId;
        }
        
        public Resource getResourceRequest() {
            return resourceRequest;
        }
        
        public long getCreationTime() {
            return creationTime;
        }
        
        public long getWaitingTime() {
            return System.currentTimeMillis() - creationTime;
        }
        
        public int getRetryCount() {
            return retryCount;
        }
        
        public void incrementRetryCount() {
            this.retryCount++;
        }
        
        @Override
        public String toString() {
            return "SchedulingTask{" +
                   "containerId=" + containerId +
                   ", applicationId=" + applicationId +
                   ", retryCount=" + retryCount +
                   ", waitingTime=" + getWaitingTime() + "ms" +
                   '}';
        }
    }
}