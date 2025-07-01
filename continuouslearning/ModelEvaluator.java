package com.perphproctor.continuouslearning;

import com.perphproctor.common.Utils;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;

/**
 * Evaluates and tracks the performance of machine learning models.
 * This class provides methods to calculate various performance metrics
 * and track model performance over time.
 */
public class ModelEvaluator implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    // Performance metrics history
    private final Queue<PerformanceMetrics> metricsHistory;
    private final int historySize;
    
    // Current performance metrics
    private PerformanceMetrics currentMetrics;
    
    // Evaluation tracking
    private long lastEvaluationTime;
    private int evaluationCount;
    
    /**
     * Constructor with default history size
     */
    public ModelEvaluator() {
        this(10);
    }
    
    /**
     * Constructor with custom history size
     * 
     * @param historySize Number of historical metrics to keep
     */
    public ModelEvaluator(int historySize) {
        this.historySize = historySize;
        this.metricsHistory = new LinkedList<>();
        this.currentMetrics = new PerformanceMetrics();
        this.lastEvaluationTime = System.currentTimeMillis();
        this.evaluationCount = 0;
    }
    
    /**
     * Evaluate a model using test data
     * 
     * @param model The model to evaluate
     * @param testFeatures Test feature vectors
     * @param testTargets Test target values
     * @return Performance metrics
     */
    public PerformanceMetrics evaluate(ContinualLearningTree model, double[][] testFeatures, double[] testTargets) {
        if (model == null || testFeatures == null || testTargets == null || 
            testFeatures.length == 0 || testFeatures.length != testTargets.length) {
            throw new IllegalArgumentException("Invalid evaluation parameters");
        }
        
        // Make predictions
        double[] predictions = new double[testTargets.length];
        for (int i = 0; i < testFeatures.length; i++) {
            predictions[i] = model.predict(testFeatures[i]);
        }
        
        // Calculate performance metrics
        PerformanceMetrics metrics = calculateMetrics(predictions, testTargets);
        
        // Update history
        updateHistory(metrics);
        
        return metrics;
    }
    
    /**
     * Evaluate a forest model using test data
     * 
     * @param model The forest model to evaluate
     * @param testFeatures Test feature vectors
     * @param testTargets Test target values
     * @return Performance metrics
     */
    public PerformanceMetrics evaluate(ContinualLearningRandomForest model, double[][] testFeatures, double[] testTargets) {
        if (model == null || testFeatures == null || testTargets == null || 
            testFeatures.length == 0 || testFeatures.length != testTargets.length) {
            throw new IllegalArgumentException("Invalid evaluation parameters");
        }
        
        // Make predictions
        double[] predictions = new double[testTargets.length];
        for (int i = 0; i < testFeatures.length; i++) {
            predictions[i] = model.predict(testFeatures[i]);
        }
        
        // Calculate performance metrics
        PerformanceMetrics metrics = calculateMetrics(predictions, testTargets);
        
        // Update history
        updateHistory(metrics);
        
        return metrics;
    }
    
    /**
     * Calculate performance metrics from predictions and actual values
     * 
     * @param predictions Predicted values
     * @param actuals Actual values
     * @return Performance metrics
     */
    public PerformanceMetrics calculateMetrics(double[] predictions, double[] actuals) {
        if (predictions == null || actuals == null || predictions.length != actuals.length) {
            throw new IllegalArgumentException("Invalid arrays for metric calculation");
        }
        
        int n = predictions.length;
        
        // Calculate basic metrics
        double rmse = Utils.calculateRMSE(predictions, actuals);
        double mae = Utils.calculateMAE(predictions, actuals);
        double r2 = Utils.calculateR2(predictions, actuals);
        double mape = Utils.calculateMAPE(predictions, actuals);
        
        // Calculate error distributions
        double[] errors = new double[n];
        double[] absErrors = new double[n];
        double[] relErrors = new double[n];
        
        for (int i = 0; i < n; i++) {
            errors[i] = predictions[i] - actuals[i];
            absErrors[i] = Math.abs(errors[i]);
            if (actuals[i] != 0) {
                relErrors[i] = Math.abs(errors[i] / actuals[i]);
            } else {
                relErrors[i] = 0;
            }
        }
        
        // Sort for percentile calculations
        Arrays.sort(absErrors);
        Arrays.sort(relErrors);
        
        // Calculate error percentiles
        double absErrorP50 = absErrors[(int)(n * 0.5)];
        double absErrorP90 = absErrors[(int)(n * 0.9)];
        double absErrorP95 = absErrors[(int)(n * 0.95)];
        
        double relErrorP50 = relErrors[(int)(n * 0.5)];
        double relErrorP90 = relErrors[(int)(n * 0.9)];
        double relErrorP95 = relErrors[(int)(n * 0.95)];
        
        // Create and return metrics object
        PerformanceMetrics metrics = new PerformanceMetrics();
        metrics.rmse = rmse;
        metrics.mae = mae;
        metrics.r2 = r2;
        metrics.mape = mape;
        metrics.sampleCount = n;
        metrics.absErrorP50 = absErrorP50;
        metrics.absErrorP90 = absErrorP90;
        metrics.absErrorP95 = absErrorP95;
        metrics.relErrorP50 = relErrorP50;
        metrics.relErrorP90 = relErrorP90;
        metrics.relErrorP95 = relErrorP95;
        metrics.timestamp = System.currentTimeMillis();
        
        return metrics;
    }
    
    /**
     * Update metrics history with new metrics
     * 
     * @param metrics New performance metrics
     */
    private void updateHistory(PerformanceMetrics metrics) {
        // Add to history
        metricsHistory.add(metrics);
        
        // Remove oldest if history exceeds size limit
        while (metricsHistory.size() > historySize) {
            metricsHistory.poll();
        }
        
        // Update current metrics
        currentMetrics = metrics;
        lastEvaluationTime = System.currentTimeMillis();
        evaluationCount++;
    }
    
    /**
     * Get the current performance metrics
     * 
     * @return Current performance metrics
     */
    public PerformanceMetrics getCurrentMetrics() {
        return currentMetrics;
    }
    
    /**
     * Get the complete metrics history
     * 
     * @return Array of historical performance metrics
     */
    public PerformanceMetrics[] getMetricsHistory() {
        return metricsHistory.toArray(new PerformanceMetrics[0]);
    }
    
    /**
     * Check if model performance has degraded significantly
     * 
     * @param degradationThreshold Threshold for performance degradation
     * @return true if performance has degraded beyond the threshold
     */
    public boolean hasPerformanceDegraded(double degradationThreshold) {
        if (metricsHistory.size() < 2) {
            return false;
        }
        
        PerformanceMetrics[] history = getMetricsHistory();
        
        // Calculate average historical performance (excluding current)
        double avgHistoricalRMSE = 0.0;
        int count = 0;
        
        for (int i = 0; i < history.length - 1; i++) {
            avgHistoricalRMSE += history[i].rmse;
            count++;
        }
        
        if (count > 0) {
            avgHistoricalRMSE /= count;
            
            // Check if current performance is worse by threshold percentage
            double currentRMSE = currentMetrics.rmse;
            double degradation = (currentRMSE - avgHistoricalRMSE) / avgHistoricalRMSE;
            
            return degradation > degradationThreshold;
        }
        
        return false;
    }
    
    /**
     * Calculate performance trend over time
     * 
     * @return Map of metric names to trend values (positive = improving, negative = degrading)
     */
    public Map<String, Double> calculateTrends() {
        if (metricsHistory.size() < 2) {
            return new HashMap<>();
        }
        
        PerformanceMetrics[] history = getMetricsHistory();
        Map<String, Double> trends = new HashMap<>();
        
        // Calculate linear regression slope for each metric
        double[] timestamps = new double[history.length];
        double[] rmseValues = new double[history.length];
        double[] maeValues = new double[history.length];
        double[] r2Values = new double[history.length];
        
        for (int i = 0; i < history.length; i++) {
            timestamps[i] = (double)(history[i].timestamp - history[0].timestamp) / 1000.0; // Convert to seconds
            rmseValues[i] = history[i].rmse;
            maeValues[i] = history[i].mae;
            r2Values[i] = history[i].r2;
        }
        
        // Calculate slopes
        double rmseSlope = calculateSlope(timestamps, rmseValues);
        double maeSlope = calculateSlope(timestamps, maeValues);
        double r2Slope = calculateSlope(timestamps, r2Values);
        
        // For RMSE and MAE, negative slope means improvement
        trends.put("RMSE", -rmseSlope);
        trends.put("MAE", -maeSlope);
        // For R², positive slope means improvement
        trends.put("R2", r2Slope);
        
        return trends;
    }
    
    /**
     * Calculate the slope of a linear regression line
     * 
     * @param x X values (independent variable)
     * @param y Y values (dependent variable)
     * @return Slope of the regression line
     */
    private double calculateSlope(double[] x, double[] y) {
        if (x.length != y.length || x.length < 2) {
            return 0.0;
        }
        
        int n = x.length;
        double sumX = 0.0;
        double sumY = 0.0;
        double sumXY = 0.0;
        double sumX2 = 0.0;
        
        for (int i = 0; i < n; i++) {
            sumX += x[i];
            sumY += y[i];
            sumXY += x[i] * y[i];
            sumX2 += x[i] * x[i];
        }
        
        double denominator = n * sumX2 - sumX * sumX;
        if (Math.abs(denominator) < 1e-10) {
            return 0.0; // Avoid division by zero
        }
        
        return (n * sumXY - sumX * sumY) / denominator;
    }
    
    /**
     * Get the time of the last evaluation
     * 
     * @return Timestamp of last evaluation
     */
    public long getLastEvaluationTime() {
        return lastEvaluationTime;
    }
    
    /**
     * Get the number of evaluations performed
     * 
     * @return Evaluation count
     */
    public int getEvaluationCount() {
        return evaluationCount;
    }
    
    /**
     * Check if an evaluation is needed based on elapsed time
     * 
     * @param intervalMs Minimum time interval between evaluations
     * @return true if an evaluation is needed
     */
    public boolean isEvaluationNeeded(long intervalMs) {
        return System.currentTimeMillis() - lastEvaluationTime >= intervalMs;
    }
    
    /**
     * Reset the evaluator's state
     */
    public void reset() {
        metricsHistory.clear();
        currentMetrics = new PerformanceMetrics();
        lastEvaluationTime = System.currentTimeMillis();
        evaluationCount = 0;
    }
    
    /**
     * Class representing a set of performance metrics
     */
    public static class PerformanceMetrics implements Serializable {
        private static final long serialVersionUID = 1L;
        
        // Basic metrics
        public double rmse;  // Root Mean Square Error
        public double mae;   // Mean Absolute Error
        public double r2;    // R-squared (coefficient of determination)
        public double mape;  // Mean Absolute Percentage Error
        
        // Error distribution metrics
        public double absErrorP50;  // 50th percentile of absolute errors (median)
        public double absErrorP90;  // 90th percentile of absolute errors
        public double absErrorP95;  // 95th percentile of absolute errors
        
        public double relErrorP50;  // 50th percentile of relative errors
        public double relErrorP90;  // 90th percentile of relative errors
        public double relErrorP95;  // 95th percentile of relative errors
        
        // Metadata
        public int sampleCount;      // Number of samples used for evaluation
        public long timestamp;       // When these metrics were calculated
        
        /**
         * Default constructor
         */
        public PerformanceMetrics() {
            this.rmse = 0.0;
            this.mae = 0.0;
            this.r2 = 0.0;
            this.mape = 0.0;
            this.absErrorP50 = 0.0;
            this.absErrorP90 = 0.0;
            this.absErrorP95 = 0.0;
            this.relErrorP50 = 0.0;
            this.relErrorP90 = 0.0;
            this.relErrorP95 = 0.0;
            this.sampleCount = 0;
            this.timestamp = System.currentTimeMillis();
        }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("PerformanceMetrics[");
            sb.append("rmse=").append(String.format("%.4f", rmse));
            sb.append(", mae=").append(String.format("%.4f", mae));
            sb.append(", r2=").append(String.format("%.4f", r2));
            sb.append(", mape=").append(String.format("%.2f", mape)).append("%");
            sb.append(", samples=").append(sampleCount);
            sb.append(", p95AbsErr=").append(String.format("%.4f", absErrorP95));
            sb.append(", p95RelErr=").append(String.format("%.2f", relErrorP95 * 100)).append("%");
            sb.append("]");
            return sb.toString();
        }
        
        /**
         * Get a formatted summary of the metrics
         * 
         * @return Formatted metrics summary
         */
        public String getFormattedSummary() {
            StringBuilder sb = new StringBuilder();
            sb.append("Model Performance Metrics:\n");
            sb.append("------------------------\n");
            sb.append(String.format("RMSE: %.4f\n", rmse));
            sb.append(String.format("MAE: %.4f\n", mae));
            sb.append(String.format("R²: %.4f\n", r2));
            sb.append(String.format("MAPE: %.2f%%\n", mape));
            sb.append(String.format("Sample Count: %d\n", sampleCount));
            sb.append("\nError Distributions:\n");
            sb.append("------------------\n");
            sb.append(String.format("Absolute Error (Median): %.4f\n", absErrorP50));
            sb.append(String.format("Absolute Error (90th Percentile): %.4f\n", absErrorP90));
            sb.append(String.format("Absolute Error (95th Percentile): %.4f\n", absErrorP95));
            sb.append(String.format("Relative Error (Median): %.2f%%\n", relErrorP50 * 100));
            sb.append(String.format("Relative Error (90th Percentile): %.2f%%\n", relErrorP90 * 100));
            sb.append(String.format("Relative Error (95th Percentile): %.2f%%\n", relErrorP95 * 100));
            return sb.toString();
        }
    }
}