package com.perphproctor.continuouslearning;

import com.perphproctor.common.Constants;
import com.perphproctor.common.Utils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Implementation of a Classification and Regression Tree (CART) with 
 * continual learning capabilities for the PerphProctor system.
 * 
 * This tree can be incrementally updated with new data samples without
 * requiring complete retraining, enabling efficient adaptation to 
 * changing workload patterns.
 */
public class ContinualLearningTree implements Serializable {
    
    private static final long serialVersionUID = 1L;
    private static final Random random = new Random();
    
    // Tree node structure
    private Node root;
    
    // Tree metadata
    private int featureCount;
    private int nodeCount;
    private long lastUpdateTimestamp;
    private int updateCount;
    
    // Configuration parameters
    private final int minSamplesForSplit;
    private final double maxRssThreshold;
    
    /**
     * Constructor with default configuration parameters
     * 
     * @param featureCount Number of features in the input data
     */
    public ContinualLearningTree(int featureCount) {
        this(featureCount, Constants.MIN_SAMPLES_FOR_SPLIT, Constants.MAX_RSS_THRESHOLD);
    }
    
    /**
     * Constructor with custom configuration parameters
     * 
     * @param featureCount Number of features in the input data
     * @param minSamplesForSplit Minimum number of samples required to split a node
     * @param maxRssThreshold Maximum allowed RSS for a leaf node
     */
    public ContinualLearningTree(int featureCount, int minSamplesForSplit, double maxRssThreshold) {
        this.featureCount = featureCount;
        this.minSamplesForSplit = minSamplesForSplit;
        this.maxRssThreshold = maxRssThreshold;
        this.nodeCount = 0;
        this.updateCount = 0;
        this.lastUpdateTimestamp = System.currentTimeMillis();
        this.root = null;
    }
    
    /**
     * Train the tree with a set of samples
     * 
     * @param X Feature vectors for training
     * @param y Target values for training
     * @return This tree instance for method chaining
     */
    public ContinualLearningTree train(double[][] X, double[] y) {
        if (X == null || y == null || X.length == 0 || X.length != y.length) {
            throw new IllegalArgumentException("Invalid training data");
        }
        
        if (X[0].length != featureCount) {
            throw new IllegalArgumentException("Feature count mismatch");
        }
        
        // Create root node with all training samples
        List<Integer> sampleIndices = new ArrayList<>(X.length);
        for (int i = 0; i < X.length; i++) {
            sampleIndices.add(i);
        }
        
        root = buildTree(X, y, sampleIndices, 0);
        lastUpdateTimestamp = System.currentTimeMillis();
        return this;
    }
    
    /**
     * Update the tree with a new sample
     * 
     * @param x Feature vector for the new sample
     * @param y Target value for the new sample
     * @return This tree instance for method chaining
     */
    public ContinualLearningTree update(double[] x, double y) {
        if (x == null || x.length != featureCount) {
            throw new IllegalArgumentException("Invalid feature vector");
        }
        
        if (root == null) {
            // If tree is empty, create a new leaf node with the sample
            root = new Node();
            root.isLeaf = true;
            root.prediction = y;
            root.samples = new ArrayList<>();
            root.samples.add(0);
            nodeCount = 1;
        } else {
            // Find the leaf node where the sample belongs
            Node leaf = findLeafNode(root, x);
            
            // Update the leaf node with the new sample
            updateLeafNode(leaf, x, y);
        }
        
        updateCount++;
        lastUpdateTimestamp = System.currentTimeMillis();
        return this;
    }
    
    /**
     * Predict the target value for a feature vector
     * 
     * @param x Feature vector
     * @return Predicted target value
     */
    public double predict(double[] x) {
        if (x == null || x.length != featureCount) {
            throw new IllegalArgumentException("Invalid feature vector");
        }
        
        if (root == null) {
            return 0.0; // Default prediction for empty tree
        }
        
        return predictRecursive(root, x);
    }
    
    /**
     * Get the number of nodes in the tree
     * 
     * @return Node count
     */
    public int getNodeCount() {
        return nodeCount;
    }
    
    /**
     * Get the timestamp of the last update
     * 
     * @return Timestamp in milliseconds
     */
    public long getLastUpdateTimestamp() {
        return lastUpdateTimestamp;
    }
    
    /**
     * Get the number of updates made to the tree
     * 
     * @return Update count
     */
    public int getUpdateCount() {
        return updateCount;
    }
    
    /**
     * Find the leaf node where a sample belongs
     * 
     * @param node Current node in traversal
     * @param x Feature vector
     * @return Leaf node
     */
    private Node findLeafNode(Node node, double[] x) {
        if (node.isLeaf) {
            return node;
        }
        
        if (x[node.featureIndex] <= node.threshold) {
            return findLeafNode(node.left, x);
        } else {
            return findLeafNode(node.right, x);
        }
    }
    
    /**
     * Update a leaf node with a new sample and potentially split it
     * 
     * @param leaf Leaf node to update
     * @param x Feature vector
     * @param y Target value
     */
    private void updateLeafNode(Node leaf, double[] x, double y) {
        // Add sample to node's samples
        int sampleIndex = leaf.samples.size();
        leaf.samples.add(sampleIndex);
        
        // Store sample data for use in potential split
        if (leaf.sampleFeatures == null) {
            leaf.sampleFeatures = new ArrayList<>();
            leaf.sampleTargets = new ArrayList<>();
        }
        
        leaf.sampleFeatures.add(x);
        leaf.sampleTargets.add(y);
        
        // Update the prediction (mean of all targets)
        double sum = 0.0;
        for (Double target : leaf.sampleTargets) {
            sum += target;
        }
        leaf.prediction = sum / leaf.sampleTargets.size();
        
        // Check if node should be split
        if (leaf.sampleFeatures.size() >= minSamplesForSplit) {
            // Calculate RSS
            double rss = calculateRSS(leaf.sampleTargets, leaf.prediction);
            
            // If RSS exceeds threshold, attempt to split
            if (rss > maxRssThreshold) {
                splitLeafNode(leaf);
            }
        }
    }
    
    /**
     * Calculate the Residual Sum of Squares for a set of target values
     * 
     * @param targets List of target values
     * @param prediction Prediction value
     * @return RSS value
     */
    private double calculateRSS(List<Double> targets, double prediction) {
        double rss = 0.0;
        for (Double target : targets) {
            double residual = target - prediction;
            rss += residual * residual;
        }
        return rss;
    }
    
    /**
     * Split a leaf node based on the best feature and threshold
     * 
     * @param leaf Leaf node to split
     */
    private void splitLeafNode(Node leaf) {
        double bestGain = -Double.MAX_VALUE;
        int bestFeature = -1;
        double bestThreshold = 0.0;
        List<Integer> bestLeft = null;
        List<Integer> bestRight = null;
        
        int sampleCount = leaf.sampleFeatures.size();
        double currentRSS = calculateRSS(leaf.sampleTargets, leaf.prediction);
        
        // Convert lists to arrays for faster access
        double[][] X = leaf.sampleFeatures.toArray(new double[sampleCount][]);
        double[] y = new double[sampleCount];
        for (int i = 0; i < sampleCount; i++) {
            y[i] = leaf.sampleTargets.get(i);
        }
        
        // Try all features and possible thresholds
        for (int featureIndex = 0; featureIndex < featureCount; featureIndex++) {
            // Get unique values for the feature
            double[] featureValues = new double[sampleCount];
            for (int i = 0; i < sampleCount; i++) {
                featureValues[i] = X[i][featureIndex];
            }
            Arrays.sort(featureValues);
            
            // Try thresholds between consecutive unique values
            for (int i = 0; i < sampleCount - 1; i++) {
                if (featureValues[i] == featureValues[i + 1]) {
                    continue; // Skip duplicate values
                }
                
                double threshold = (featureValues[i] + featureValues[i + 1]) / 2.0;
                
                // Split samples based on threshold
                List<Integer> leftIndices = new ArrayList<>();
                List<Integer> rightIndices = new ArrayList<>();
                
                double leftSum = 0.0;
                double rightSum = 0.0;
                
                for (int j = 0; j < sampleCount; j++) {
                    if (X[j][featureIndex] <= threshold) {
                        leftIndices.add(j);
                        leftSum += y[j];
                    } else {
                        rightIndices.add(j);
                        rightSum += y[j];
                    }
                }
                
                // Skip if split doesn't meet minimum sample requirement
                if (leftIndices.size() < minSamplesForSplit / 2 || rightIndices.size() < minSamplesForSplit / 2) {
                    continue;
                }
                
                // Calculate means for left and right splits
                double leftMean = leftSum / leftIndices.size();
                double rightMean = rightSum / rightIndices.size();
                
                // Calculate RSS for left and right splits
                double leftRSS = 0.0;
                for (int idx : leftIndices) {
                    double residual = y[idx] - leftMean;
                    leftRSS += residual * residual;
                }
                
                double rightRSS = 0.0;
                for (int idx : rightIndices) {
                    double residual = y[idx] - rightMean;
                    rightRSS += residual * residual;
                }
                
                // Calculate information gain
                double splitRSS = leftRSS + rightRSS;
                double gain = currentRSS - splitRSS;
                
                // Update best split if this one is better
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeature = featureIndex;
                    bestThreshold = threshold;
                    bestLeft = leftIndices;
                    bestRight = rightIndices;
                }
            }
        }
        
        // If no valid split found or gain is minimal, keep as leaf
        if (bestFeature == -1 || bestGain < maxRssThreshold / 10) {
            return;
        }
        
        // Create child nodes
        leaf.isLeaf = false;
        leaf.featureIndex = bestFeature;
        leaf.threshold = bestThreshold;
        
        // Left child
        leaf.left = new Node();
        leaf.left.isLeaf = true;
        leaf.left.samples = bestLeft;
        
        double leftSum = 0.0;
        leaf.left.sampleFeatures = new ArrayList<>();
        leaf.left.sampleTargets = new ArrayList<>();
        
        for (int idx : bestLeft) {
            leaf.left.sampleFeatures.add(leaf.sampleFeatures.get(idx));
            leaf.left.sampleTargets.add(leaf.sampleTargets.get(idx));
            leftSum += leaf.sampleTargets.get(idx);
        }
        
        leaf.left.prediction = leftSum / bestLeft.size();
        
        // Right child
        leaf.right = new Node();
        leaf.right.isLeaf = true;
        leaf.right.samples = bestRight;
        
        double rightSum = 0.0;
        leaf.right.sampleFeatures = new ArrayList<>();
        leaf.right.sampleTargets = new ArrayList<>();
        
        for (int idx : bestRight) {
            leaf.right.sampleFeatures.add(leaf.sampleFeatures.get(idx));
            leaf.right.sampleTargets.add(leaf.sampleTargets.get(idx));
            rightSum += leaf.sampleTargets.get(idx);
        }
        
        leaf.right.prediction = rightSum / bestRight.size();
        
        // Clear sample data from parent node to save memory
        leaf.sampleFeatures = null;
        leaf.sampleTargets = null;
        
        // Update node count
        nodeCount += 2;
    }
    
    /**
     * Recursively build a tree from training samples
     * 
     * @param X Feature vectors
     * @param y Target values
     * @param sampleIndices Indices of samples to use
     * @param depth Current depth in the tree
     * @return Root node of the built subtree
     */
    private Node buildTree(double[][] X, double[] y, List<Integer> sampleIndices, int depth) {
        Node node = new Node();
        node.samples = sampleIndices;
        nodeCount++;
        
        // Calculate the mean target value for this node
        double sum = 0.0;
        for (int idx : sampleIndices) {
            sum += y[idx];
        }
        node.prediction = sum / sampleIndices.size();
        
        // Check if we should make this a leaf node
        if (sampleIndices.size() < minSamplesForSplit) {
            node.isLeaf = true;
            return node;
        }
        
        // Calculate current RSS
        double currentRSS = 0.0;
        for (int idx : sampleIndices) {
            double residual = y[idx] - node.prediction;
            currentRSS += residual * residual;
        }
        
        // If RSS is small enough, make this a leaf node
        if (currentRSS <= maxRssThreshold) {
            node.isLeaf = true;
            return node;
        }
        
        // Find the best split
        double bestGain = -Double.MAX_VALUE;
        int bestFeature = -1;
        double bestThreshold = 0.0;
        List<Integer> bestLeft = null;
        List<Integer> bestRight = null;
        
        // Randomly select sqrt(featureCount) features to consider
        int featuresToConsider = Math.max(1, (int) Math.sqrt(featureCount));
        int[] featureIndices = Utils.sampleIndicesWithoutReplacement(featuresToConsider, featureCount);
        
        for (int featureIdx : featureIndices) {
            // Get values for this feature
            List<Double> featureValues = new ArrayList<>();
            for (int idx : sampleIndices) {
                featureValues.add(X[idx][featureIdx]);
            }
            
            // Sort unique feature values
            featureValues.sort(null);
            
            // Try thresholds between consecutive unique values
            for (int i = 0; i < featureValues.size() - 1; i++) {
                if (featureValues.get(i).equals(featureValues.get(i + 1))) {
                    continue; // Skip duplicate values
                }
                
                double threshold = (featureValues.get(i) + featureValues.get(i + 1)) / 2.0;
                
                // Split samples based on threshold
                List<Integer> leftIndices = new ArrayList<>();
                List<Integer> rightIndices = new ArrayList<>();
                
                for (int idx : sampleIndices) {
                    if (X[idx][featureIdx] <= threshold) {
                        leftIndices.add(idx);
                    } else {
                        rightIndices.add(idx);
                    }
                }
                
                // Skip if split doesn't meet minimum sample requirement
                if (leftIndices.size() < minSamplesForSplit / 2 || rightIndices.size() < minSamplesForSplit / 2) {
                    continue;
                }
                
                // Calculate means for left and right splits
                double leftSum = 0.0;
                for (int idx : leftIndices) {
                    leftSum += y[idx];
                }
                double leftMean = leftSum / leftIndices.size();
                
                double rightSum = 0.0;
                for (int idx : rightIndices) {
                    rightSum += y[idx];
                }
                double rightMean = rightSum / rightIndices.size();
                
                // Calculate RSS for left and right splits
                double leftRSS = 0.0;
                for (int idx : leftIndices) {
                    double residual = y[idx] - leftMean;
                    leftRSS += residual * residual;
                }
                
                double rightRSS = 0.0;
                for (int idx : rightIndices) {
                    double residual = y[idx] - rightMean;
                    rightRSS += residual * residual;
                }
                
                // Calculate information gain
                double splitRSS = leftRSS + rightRSS;
                double gain = currentRSS - splitRSS;
                
                // Update best split if this one is better
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeature = featureIdx;
                    bestThreshold = threshold;
                    bestLeft = leftIndices;
                    bestRight = rightIndices;
                }
            }
        }
        
        // If no valid split found or gain is minimal, make this a leaf node
        if (bestFeature == -1 || bestGain < maxRssThreshold / 10) {
            node.isLeaf = true;
            return node;
        }
        
        // Create split node
        node.isLeaf = false;
        node.featureIndex = bestFeature;
        node.threshold = bestThreshold;
        
        // Recursively build left and right subtrees
        node.left = buildTree(X, y, bestLeft, depth + 1);
        node.right = buildTree(X, y, bestRight, depth + 1);
        
        return node;
    }
    
    /**
     * Recursively predict the target value for a feature vector
     * 
     * @param node Current node in traversal
     * @param x Feature vector
     * @return Predicted target value
     */
    private double predictRecursive(Node node, double[] x) {
        if (node.isLeaf) {
            return node.prediction;
        }
        
        if (x[node.featureIndex] <= node.threshold) {
            return predictRecursive(node.left, x);
        } else {
            return predictRecursive(node.right, x);
        }
    }
    
    /**
     * Inner class representing a node in the tree
     */
    private static class Node implements Serializable {
        private static final long serialVersionUID = 1L;
        
        // Node type
        boolean isLeaf = false;
        
        // Split criteria (for non-leaf nodes)
        int featureIndex;
        double threshold;
        
        // Child nodes (for non-leaf nodes)
        Node left;
        Node right;
        
        // Prediction value (for leaf nodes)
        double prediction;
        
        // Sample indices that reached this node during training
        List<Integer> samples;
        
        // Sample data for incremental updates (only in leaf nodes)
        List<double[]> sampleFeatures;
        List<Double> sampleTargets;
    }
}