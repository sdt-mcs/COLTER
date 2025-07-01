package com.perphproctor.scheduler;

import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.api.records.Container;
import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.api.records.NodeId;
import org.apache.hadoop.yarn.api.records.Priority;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.server.resourcemanager.RMContext;
import org.apache.hadoop.yarn.server.resourcemanager.rmapp.RMApp;
import org.apache.hadoop.yarn.server.resourcemanager.rmcontainer.RMContainer;
import org.apache.hadoop.yarn.server.resourcemanager.rmcontainer.RMContainerState;
import org.apache.hadoop.yarn.server.resourcemanager.scheduler.AbstractYarnScheduler;
import org.apache.hadoop.yarn.server.resourcemanager.scheduler.SchedulerApplication;
import org.apache.hadoop.yarn.server.resourcemanager.scheduler.SchedulerNode;
import org.apache.hadoop.yarn.server.resourcemanager.scheduler.common.fica.FiCaSchedulerApp;
import org.apache.hadoop.yarn.server.resourcemanager.scheduler.common.fica.FiCaSchedulerNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * TaskPlacementManager is responsible for managing task placement decisions.
 * It coordinates with the recommendation calculator to make placement decisions
 * that minimize interference between co-located workloads.
 */
public class TaskPlacementManager {
    
    private static final Logger LOG = LoggerFactory.getLogger(TaskPlacementManager.class);
    
    // Reference to main scheduler
    private final AdaptiveScheduler adaptiveScheduler;
    
    // Track LRA applications
    private final Set<ApplicationId> lraApplications;
    
    // Track batch applications
    private final Set<ApplicationId> batchApplications;
    
    // Track node LRA density
    private final Map<NodeId, Integer> nodeLraCounts;
    
    // Track data locality information
    private final Map<ContainerId, List<NodeId>> dataLocalityMap;
    
    /**
     * Constructor
     * 
     * @param adaptiveScheduler Reference to the main adaptive scheduler
     */
    public TaskPlacementManager(AdaptiveScheduler adaptiveScheduler) {
        this.adaptiveScheduler = adaptiveScheduler;
        this.lraApplications = Collections.newSetFromMap(new ConcurrentHashMap<>());
        this.batchApplications = Collections.newSetFromMap(new ConcurrentHashMap<>());
        this.nodeLraCounts = new ConcurrentHashMap<>();
        this.dataLocalityMap = new ConcurrentHashMap<>();
        
        LOG.info("TaskPlacementManager initialized");
    }
    
    /**
     * Register an application as an LRA
     * 
     * @param applicationId Application ID
     */
    public void registerLraApplication(ApplicationId applicationId) {
        if (applicationId != null) {
            lraApplications.add(applicationId);
            LOG.info("Registered LRA application: {}", applicationId);
        }
    }
    
    /**
     * Register an application as a batch job
     * 
     * @param applicationId Application ID
     */
    public void registerBatchApplication(ApplicationId applicationId) {
        if (applicationId != null) {
            batchApplications.add(applicationId);
            LOG.info("Registered batch application: {}", applicationId);
        }
    }
    
    /**
     * Check if an application is an LRA
     * 
     * @param applicationId Application ID
     * @return true if application is an LRA
     */
    public boolean isLraApplication(ApplicationId applicationId) {
        return applicationId != null && lraApplications.contains(applicationId);
    }
    
    /**
     * Check if an application is a batch job
     * 
     * @param applicationId Application ID
     * @return true if application is a batch job
     */
    public boolean isBatchApplication(ApplicationId applicationId) {
        return applicationId != null && batchApplications.contains(applicationId);
    }
    
    /**
     * Update the LRA count for a node
     * 
     * @param nodeId Node ID
     * @param delta Change in LRA count (+1 for addition, -1 for removal)
     */
    public void updateNodeLraCount(NodeId nodeId, int delta) {
        if (nodeId == null) {
            return;
        }
        
        nodeLraCounts.compute(nodeId, (key, count) -> {
            if (count == null) {
                return Math.max(0, delta);
            } else {
                return Math.max(0, count + delta);
            }
        });
    }
    
    /**
     * Get the LRA count for a node
     * 
     * @param nodeId Node ID
     * @return LRA count
     */
    public int getNodeLraCount(NodeId nodeId) {
        if (nodeId == null) {
            return 0;
        }
        
        return nodeLraCounts.getOrDefault(nodeId, 0);
    }
    
    /**
     * Get all nodes with their LRA counts
     * 
     * @return Map of node IDs to LRA counts
     */
    public Map<NodeId, Integer> getAllNodeLraCounts() {
        return new HashMap<>(nodeLraCounts);
    }
    
    /**
     * Register data locality information for a container
     * 
     * @param containerId Container ID
     * @param nodes List of nodes with data locality
     */
    public void registerDataLocality(ContainerId containerId, List<NodeId> nodes) {
        if (containerId != null && nodes != null && !nodes.isEmpty()) {
            dataLocalityMap.put(containerId, new ArrayList<>(nodes));
        }
    }
    
    /**
     * Check if a node has data locality for a container
     * 
     * @param containerId Container ID
     * @param nodeId Node ID
     * @return true if node has data locality
     */
    public boolean hasDataLocality(ContainerId containerId, NodeId nodeId) {
        if (containerId == null || nodeId == null) {
            return false;
        }
        
        List<NodeId> localNodes = dataLocalityMap.get(containerId);
        return localNodes != null && localNodes.contains(nodeId);
    }
    
    /**
     * Get nodes with data locality for a container
     * 
     * @param containerId Container ID
     * @return List of nodes with data locality
     */
    public List<NodeId> getLocalityNodes(ContainerId containerId) {
        if (containerId == null) {
            return Collections.emptyList();
        }
        
        List<NodeId> localNodes = dataLocalityMap.get(containerId);
        return localNodes != null ? new ArrayList<>(localNodes) : Collections.emptyList();
    }
    
    /**
     * Clear data locality information for a container
     * 
     * @param containerId Container ID
     */
    public void clearDataLocality(ContainerId containerId) {
        if (containerId != null) {
            dataLocalityMap.remove(containerId);
        }
    }
    
    /**
     * Find candidate nodes for scheduling a container
     * 
     * @param containerId Container ID
     * @param resource Resource requirements
     * @return List of candidate nodes sorted by suitability
     */
    public List<SchedulerNode> findCandidateNodes(ContainerId containerId, Resource resource) {
        AbstractYarnScheduler scheduler = adaptiveScheduler.getScheduler();
        List<SchedulerNode> candidateNodes = new ArrayList<>();
        
        if (scheduler == null || containerId == null || resource == null) {
            return candidateNodes;
        }
        
        ApplicationId applicationId = containerId.getApplicationAttemptId().getApplicationId();
        
        // Get all nodes from the scheduler
        for (NodeId nodeId : scheduler.getNodeTracker().getNodeIds()) {
            SchedulerNode node = scheduler.getNodeTracker().getNode(nodeId);
            
            if (node == null) {
                continue;
            }
            
            // Check if node has enough resources
            Resource available = node.getAvailableResource();
            if (available.getMemorySize() < resource.getMemorySize() ||
                available.getVirtualCores() < resource.getVirtualCores()) {
                continue;
            }
            
            // Add to candidates
            candidateNodes.add(node);
        }
        
        // Sort candidates
        if (isBatchApplication(applicationId)) {
            // For batch jobs, sort by recommendation score (highest first)
            candidateNodes.sort((a, b) -> {
                double scoreA = adaptiveScheduler.getNodeRecommendationScores()
                                              .getOrDefault(a.getNodeID(), 0.5);
                double scoreB = adaptiveScheduler.getNodeRecommendationScores()
                                              .getOrDefault(b.getNodeID(), 0.5);
                return Double.compare(scoreB, scoreA);
            });
        } else {
            // For LRA applications, sort by LRA count (lowest first)
            candidateNodes.sort(Comparator.comparingInt(
                    a -> getNodeLraCount(a.getNodeID())));
        }
        
        return candidateNodes;
    }
    
    /**
     * Calculate node suitability score for a container
     * 
     * @param containerId Container ID
     * @param nodeId Node ID
     * @return Suitability score (higher is better)
     */
    public double calculateNodeSuitabilityScore(ContainerId containerId, NodeId nodeId) {
        if (containerId == null || nodeId == null) {
            return 0.0;
        }
        
        ApplicationId applicationId = containerId.getApplicationAttemptId().getApplicationId();
        
        // Base score
        double score = 0.5;
        
        // Application type adjustment
        if (isBatchApplication(applicationId)) {
            // For batch jobs, use recommendation score
            score = adaptiveScheduler.getNodeRecommendationScores()
                                   .getOrDefault(nodeId, 0.5);
        } else {
            // For LRAs, prefer nodes with fewer LRAs
            int lraCount = getNodeLraCount(nodeId);
            score = 1.0 / (1.0 + lraCount);
        }
        
        // Data locality bonus
        if (hasDataLocality(containerId, nodeId)) {
            score += 0.3;
        }
        
        return score;
    }
    
    /**
     * Attempt to place a container on the best suitable node
     * 
     * @param container Container to place
     * @param resource Resource requirements
     * @return Selected node or null if no suitable node found
     */
    public SchedulerNode selectNodeForContainer(Container container, Resource resource) {
        if (container == null || resource == null) {
            return null;
        }
        
        ContainerId containerId = container.getId();
        ApplicationId applicationId = containerId.getApplicationAttemptId().getApplicationId();
        
        // Get candidate nodes
        List<SchedulerNode> candidates = findCandidateNodes(containerId, resource);
        
        // Check if we have any candidates
        if (candidates.isEmpty()) {
            return null;
        }
        
        // For batch applications, apply QoS-aware scheduling
        if (isBatchApplication(applicationId)) {
            for (SchedulerNode node : candidates) {
                // Check if node accepts the task
                if (adaptiveScheduler.canNodeAcceptTask(node.getNodeID()) ||
                    adaptiveScheduler.shouldScheduleAnyway(containerId, node.getNodeID())) {
                    return node;
                }
            }
            
            // If no node accepted the task with QoS constraints, return null
            return null;
        } else {
            // For LRA applications, select the first (best) candidate
            return candidates.get(0);
        }
    }
    
    /**
     * Process container allocation
     * 
     * @param container Allocated container
     * @param nodeId Node where container was allocated
     */
    public void processContainerAllocation(Container container, NodeId nodeId) {
        if (container == null || nodeId == null) {
            return;
        }
        
        ContainerId containerId = container.getId();
        ApplicationId applicationId = containerId.getApplicationAttemptId().getApplicationId();
        
        // Update node LRA count if this is an LRA
        if (isLraApplication(applicationId)) {
            updateNodeLraCount(nodeId, 1);
            LOG.info("LRA container {} allocated on node {}", containerId, nodeId);
        } else if (isBatchApplication(applicationId)) {
            LOG.info("Batch container {} allocated on node {}", containerId, nodeId);
        }
    }
    
    /**
     * Process container completion
     * 
     * @param container Completed container
     * @param nodeId Node where container was running
     */
    public void processContainerCompletion(Container container, NodeId nodeId) {
        if (container == null || nodeId == null) {
            return;
        }
        
        ContainerId containerId = container.getId();
        ApplicationId applicationId = containerId.getApplicationAttemptId().getApplicationId();
        
        // Update node LRA count if this was an LRA
        if (isLraApplication(applicationId)) {
            updateNodeLraCount(nodeId, -1);
            LOG.info("LRA container {} completed on node {}", containerId, nodeId);
        } else if (isBatchApplication(applicationId)) {
            LOG.info("Batch container {} completed on node {}", containerId, nodeId);
        }
        
        // Clear data locality information
        clearDataLocality(containerId);
        
        // Clear task retry information
        adaptiveScheduler.clearTaskInfo(containerId);
    }
    
    /**
     * Process application completion
     * 
     * @param applicationId Completed application ID
     */
    public void processApplicationCompletion(ApplicationId applicationId) {
        if (applicationId == null) {
            return;
        }
        
        // Remove from application sets
        lraApplications.remove(applicationId);
        batchApplications.remove(applicationId);
        
        LOG.info("Application {} completed", applicationId);
    }
    
    /**
     * Get statistics about current placement state
     * 
     * @return Map of statistics
     */
    public Map<String, Object> getPlacementStatistics() {
        Map<String, Object> stats = new HashMap<>();
        
        stats.put("lraApplicationCount", lraApplications.size());
        stats.put("batchApplicationCount", batchApplications.size());
        stats.put("nodeLraCounts", new HashMap<>(nodeLraCounts));
        
        // Calculate average LRA count per node
        double avgLraCount = nodeLraCounts.values().stream()
                .mapToInt(Integer::intValue)
                .average()
                .orElse(0.0);
        stats.put("averageLraCountPerNode", avgLraCount);
        
        // Calculate node recommendation score statistics
        Map<NodeId, Double> scores = adaptiveScheduler.getNodeRecommendationScores();
        double avgScore = scores.values().stream()
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0.5);
        stats.put("averageNodeScore", avgScore);
        
        double minScore = scores.values().stream()
                .mapToDouble(Double::doubleValue)
                .min()
                .orElse(0.0);
        stats.put("minNodeScore", minScore);
        
        double maxScore = scores.values().stream()
                .mapToDouble(Double::doubleValue)
                .max()
                .orElse(1.0);
        stats.put("maxNodeScore", maxScore);
        
        return stats;
    }
}