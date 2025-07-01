package com.perphproctor.common;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.OperatingSystemMXBean;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.net.InetAddress;
import java.net.UnknownHostException;

/**
 * Utility functions for the PerphProctor system.
 * This class provides various helper methods used throughout the system.
 */
public class Utils {
    
    private static final Random random = new Random();
    private static final int DEFAULT_PRECISION = 4;
    
    /**
     * Private constructor to prevent instantiation
     */
    private Utils() {
        // Private constructor to prevent instantiation
    }
    
    /**
     * Calculate the mean value of an array of doubles
     * 
     * @param values Array of double values
     * @return Mean value, or 0 if array is empty
     */
    public static double mean(double[] values) {
        if (values == null || values.length == 0) {
            return 0.0;
        }
        
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        
        return sum / values.length;
    }
    
    /**
     * Calculate the mean value of a collection of doubles
     * 
     * @param values Collection of double values
     * @return Mean value, or 0 if collection is empty
     */
    public static double mean(Collection<Double> values) {
        if (values == null || values.isEmpty()) {
            return 0.0;
        }
        
        double sum = 0.0;
        for (Double value : values) {
            sum += value;
        }
        
        return sum / values.size();
    }
    
    /**
     * Calculate the standard deviation of an array of doubles
     * 
     * @param values Array of double values
     * @return Standard deviation, or 0 if array is empty or has only one element
     */
    public static double standardDeviation(double[] values) {
        if (values == null || values.length <= 1) {
            return 0.0;
        }
        
        double mean = mean(values);
        double sumSquaredDifferences = 0.0;
        
        for (double value : values) {
            double difference = value - mean;
            sumSquaredDifferences += difference * difference;
        }
        
        return Math.sqrt(sumSquaredDifferences / (values.length - 1));
    }
    
    /**
     * Calculate the standard deviation of a collection of doubles
     * 
     * @param values Collection of double values
     * @return Standard deviation, or 0 if collection is empty or has only one element
     */
    public static double standardDeviation(Collection<Double> values) {
        if (values == null || values.size() <= 1) {
            return 0.0;
        }
        
        double mean = mean(values);
        double sumSquaredDifferences = 0.0;
        
        for (Double value : values) {
            double difference = value - mean;
            sumSquaredDifferences += difference * difference;
        }
        
        return Math.sqrt(sumSquaredDifferences / (values.size() - 1));
    }
    
    /**
     * Calculate the 95th percentile of an array of doubles
     * 
     * @param values Array of double values
     * @return 95th percentile value, or 0 if array is empty
     */
    public static double percentile95(double[] values) {
        if (values == null || values.length == 0) {
            return 0.0;
        }
        
        // Create a copy of the array to avoid modifying the original
        double[] sortedValues = Arrays.copyOf(values, values.length);
        Arrays.sort(sortedValues);
        
        // Calculate the index for the 95th percentile
        int index = (int) Math.ceil(0.95 * sortedValues.length) - 1;
        // Ensure index is within bounds
        index = Math.max(0, Math.min(sortedValues.length - 1, index));
        
        return sortedValues[index];
    }
    
    /**
     * Calculate the 95th percentile of a collection of doubles
     * 
     * @param values Collection of double values
     * @return 95th percentile value, or 0 if collection is empty
     */
    public static double percentile95(Collection<Double> values) {
        if (values == null || values.isEmpty()) {
            return 0.0;
        }
        
        // Convert collection to array and sort
        Double[] valuesArray = values.toArray(new Double[0]);
        Arrays.sort(valuesArray);
        
        // Calculate the index for the 95th percentile
        int index = (int) Math.ceil(0.95 * valuesArray.length) - 1;
        // Ensure index is within bounds
        index = Math.max(0, Math.min(valuesArray.length - 1, index));
        
        return valuesArray[index];
    }
    
    /**
     * Round a double value to a specified number of decimal places
     * 
     * @param value The value to round
     * @param places Number of decimal places
     * @return Rounded value
     */
    public static double round(double value, int places) {
        if (places < 0) {
            throw new IllegalArgumentException("Number of decimal places must be non-negative");
        }
        
        BigDecimal bd = BigDecimal.valueOf(value);
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }
    
    /**
     * Round a double value to the default precision (4 decimal places)
     * 
     * @param value The value to round
     * @return Rounded value
     */
    public static double round(double value) {
        return round(value, DEFAULT_PRECISION);
    }
    
    /**
     * Calculate the residual sum of squares (RSS) between predicted and actual values
     * 
     * @param predicted Array of predicted values
     * @param actual Array of actual values
     * @return RSS value, or Double.MAX_VALUE if arrays have different lengths
     */
    public static double calculateRSS(double[] predicted, double[] actual) {
        if (predicted == null || actual == null || predicted.length != actual.length) {
            return Double.MAX_VALUE;
        }
        
        double rss = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            double residual = predicted[i] - actual[i];
            rss += residual * residual;
        }
        
        return rss;
    }
    
    /**
     * Calculate the mean absolute error (MAE) between predicted and actual values
     * 
     * @param predicted Array of predicted values
     * @param actual Array of actual values
     * @return MAE value, or Double.MAX_VALUE if arrays have different lengths
     */
    public static double calculateMAE(double[] predicted, double[] actual) {
        if (predicted == null || actual == null || predicted.length != actual.length) {
            return Double.MAX_VALUE;
        }
        
        double sumAbsoluteError = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            sumAbsoluteError += Math.abs(predicted[i] - actual[i]);
        }
        
        return sumAbsoluteError / predicted.length;
    }
    
    /**
     * Calculate the root mean square error (RMSE) between predicted and actual values
     * 
     * @param predicted Array of predicted values
     * @param actual Array of actual values
     * @return RMSE value, or Double.MAX_VALUE if arrays have different lengths
     */
    public static double calculateRMSE(double[] predicted, double[] actual) {
        if (predicted == null || actual == null || predicted.length != actual.length) {
            return Double.MAX_VALUE;
        }
        
        double sumSquaredError = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            double error = predicted[i] - actual[i];
            sumSquaredError += error * error;
        }
        
        return Math.sqrt(sumSquaredError / predicted.length);
    }
    
    /**
     * Calculate the R-squared (coefficient of determination) between predicted and actual values
     * 
     * @param predicted Array of predicted values
     * @param actual Array of actual values
     * @return R-squared value, or Double.MIN_VALUE if arrays have different lengths
     */
    public static double calculateR2(double[] predicted, double[] actual) {
        if (predicted == null || actual == null || predicted.length != actual.length) {
            return Double.MIN_VALUE;
        }
        
        double meanActual = mean(actual);
        
        double totalSumSquares = 0.0;
        double residualSumSquares = 0.0;
        
        for (int i = 0; i < actual.length; i++) {
            double residual = predicted[i] - actual[i];
            residualSumSquares += residual * residual;
            
            double deviation = actual[i] - meanActual;
            totalSumSquares += deviation * deviation;
        }
        
        if (totalSumSquares == 0.0) {
            // If there's no variance in the actual values, R² is undefined
            return 0.0;
        }
        
        return 1 - (residualSumSquares / totalSumSquares);
    }
    
    /**
     * Generate a random integer within a specified range
     * 
     * @param min Minimum value (inclusive)
     * @param max Maximum value (exclusive)
     * @return Random integer within the specified range
     */
    public static int randomInt(int min, int max) {
        if (min >= max) {
            throw new IllegalArgumentException("Max must be greater than min");
        }
        
        return random.nextInt(max - min) + min;
    }
    
    /**
     * Generate a random double within a specified range
     * 
     * @param min Minimum value (inclusive)
     * @param max Maximum value (exclusive)
     * @return Random double within the specified range
     */
    public static double randomDouble(double min, double max) {
        if (min >= max) {
            throw new IllegalArgumentException("Max must be greater than min");
        }
        
        return min + (max - min) * random.nextDouble();
    }
    
    /**
     * Randomly sample indices without replacement
     * 
     * @param sampleSize Number of indices to sample
     * @param totalSize Total population size
     * @return Array of sampled indices
     */
    public static int[] sampleIndicesWithoutReplacement(int sampleSize, int totalSize) {
        if (sampleSize > totalSize) {
            throw new IllegalArgumentException("Sample size cannot exceed total size");
        }
        
        // Initialize array of indices
        int[] indices = new int[totalSize];
        for (int i = 0; i < totalSize; i++) {
            indices[i] = i;
        }
        
        // Shuffle the array using Fisher-Yates algorithm
        for (int i = totalSize - 1; i > totalSize - sampleSize - 1; i--) {
            int j = random.nextInt(i + 1);
            // Swap elements at i and j
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        // Return the last 'sampleSize' elements
        return Arrays.copyOfRange(indices, totalSize - sampleSize, totalSize);
    }
    
    /**
     * Load properties from a file
     * 
     * @param filePath Path to the properties file
     * @return Properties object, or null if file could not be loaded
     */
    public static Properties loadProperties(String filePath) {
        Properties properties = new Properties();
        
        try (FileInputStream fis = new FileInputStream(filePath)) {
            properties.load(fis);
            return properties;
        } catch (IOException e) {
            System.err.println("Failed to load properties file: " + filePath);
            e.printStackTrace();
            return null;
        }
    }
    
    /**
     * Save an object to a file using serialization
     * 
     * @param object Object to serialize
     * @param filePath Path to save the serialized object
     * @return true if successful, false otherwise
     */
    public static boolean saveObject(Object object, String filePath) {
        // Create parent directories if they don't exist
        File file = new File(filePath);
        File parentDir = file.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            if (!parentDir.mkdirs()) {
                System.err.println("Failed to create directory structure for: " + filePath);
                return false;
            }
        }
        
        try (FileOutputStream fos = new FileOutputStream(filePath);
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            oos.writeObject(object);
            return true;
        } catch (IOException e) {
            System.err.println("Failed to save object to file: " + filePath);
            e.printStackTrace();
            return false;
        }
    }
    
    /**
     * Load an object from a file using deserialization
     * 
     * @param filePath Path to the serialized object file
     * @return Deserialized object, or null if loading failed
     */
    public static Object loadObject(String filePath) {
        try (FileInputStream fis = new FileInputStream(filePath);
             ObjectInputStream ois = new ObjectInputStream(fis)) {
            return ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Failed to load object from file: " + filePath);
            e.printStackTrace();
            return null;
        }
    }
    
    /**
     * Get the local hostname
     * 
     * @return Local hostname, or "unknown" if it cannot be determined
     */
    public static String getHostname() {
        try {
            return InetAddress.getLocalHost().getHostName();
        } catch (UnknownHostException e) {
            return "unknown";
        }
    }
    
    /**
     * Get the current system CPU utilization
     * 
     * @return CPU utilization as a percentage (0-100)
     */
    public static double getSystemCpuUtilization() {
        OperatingSystemMXBean osBean = ManagementFactory.getOperatingSystemMXBean();
        
        // For newer JVMs that implement com.sun.management.OperatingSystemMXBean
        if (osBean instanceof com.sun.management.OperatingSystemMXBean) {
            com.sun.management.OperatingSystemMXBean sunOsBean = (com.sun.management.OperatingSystemMXBean) osBean;
            return sunOsBean.getSystemCpuLoad() * 100;
        }
        
        // Fallback: use process CPU time as an approximation
        return osBean.getSystemLoadAverage() * 100 / osBean.getAvailableProcessors();
    }
    
    /**
     * Get the current JVM memory utilization
     * 
     * @return Memory utilization as a percentage (0-100)
     */
    public static double getJvmMemoryUtilization() {
        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        long usedMemory = memoryBean.getHeapMemoryUsage().getUsed();
        long maxMemory = memoryBean.getHeapMemoryUsage().getMax();
        
        return ((double) usedMemory / maxMemory) * 100;
    }
    
    /**
     * Execute a shell command and return the output
     * 
     * @param command Command to execute
     * @return Command output as a string
     * @throws IOException If an I/O error occurs
     */
    public static String executeCommand(String command) throws IOException {
        Process process = Runtime.getRuntime().exec(command);
        
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream()))) {
            
            StringBuilder output = new StringBuilder();
            String line;
            
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }
            
            try {
                process.waitFor();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                System.err.println("Command execution interrupted: " + command);
            }
            
            return output.toString();
        }
    }
    
    /**
     * Parse a string value to a double, with a default value if parsing fails
     * 
     * @param str String to parse
     * @param defaultValue Default value to return if parsing fails
     * @return Parsed double value or default value
     */
    public static double parseDouble(String str, double defaultValue) {
        if (str == null || str.trim().isEmpty()) {
            return defaultValue;
        }
        
        try {
            return Double.parseDouble(str.trim());
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }
    
    /**
     * Parse a string value to an integer, with a default value if parsing fails
     * 
     * @param str String to parse
     * @param defaultValue Default value to return if parsing fails
     * @return Parsed integer value or default value
     */
    public static int parseInt(String str, int defaultValue) {
        if (str == null || str.trim().isEmpty()) {
            return defaultValue;
        }
        
        try {
            return Integer.parseInt(str.trim());
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }
    
    /**
     * Check if a string is null or empty
     * 
     * @param str String to check
     * @return true if the string is null or empty, false otherwise
     */
    public static boolean isNullOrEmpty(String str) {
        return str == null || str.trim().isEmpty();
    }
    
    /**
     * Split a comma-separated string into a list of strings
     * 
     * @param str Comma-separated string
     * @return List of strings, or empty list if input is null or empty
     */
    public static List<String> splitCommaSeparated(String str) {
        if (isNullOrEmpty(str)) {
            return new ArrayList<>();
        }
        
        return Arrays.stream(str.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .collect(Collectors.toList());
    }
    
    /**
     * Convert a string to a feature vector (array of doubles)
     * 
     * @param str Comma-separated string of numeric values
     * @return Array of double values, or empty array if input is invalid
     */
    public static double[] stringToFeatureVector(String str) {
        if (isNullOrEmpty(str)) {
            return new double[0];
        }
        
        String[] parts = str.split(",");
        double[] vector = new double[parts.length];
        
        for (int i = 0; i < parts.length; i++) {
            vector[i] = parseDouble(parts[i], 0.0);
        }
        
        return vector;
    }
    
    /**
     * Convert a feature vector (array of doubles) to a string
     * 
     * @param vector Array of double values
     * @return Comma-separated string of numeric values
     */
    public static String featureVectorToString(double[] vector) {
        if (vector == null || vector.length == 0) {
            return "";
        }
        
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < vector.length; i++) {
            sb.append(vector[i]);
            if (i < vector.length - 1) {
                sb.append(",");
            }
        }
        
        return sb.toString();
    }
    
    /**
     * Create all parent directories for a file path
     * 
     * @param filePath Path to file
     * @return true if directories were created or already exist, false otherwise
     */
    public static boolean createDirectories(String filePath) {
        try {
            Path path = Paths.get(filePath);
            Path parent = path.getParent();
            
            if (parent != null) {
                Files.createDirectories(parent);
            }
            
            return true;
        } catch (IOException e) {
            System.err.println("Failed to create directories for path: " + filePath);
            e.printStackTrace();
            return false;
        }
    }
    
    /**
     * Get all files in a directory with a specific extension
     * 
     * @param dirPath Directory path
     * @param extension File extension (without the dot)
     * @return List of file paths, or empty list if directory doesn't exist
     */
    public static List<String> getFilesWithExtension(String dirPath, String extension) {
        File dir = new File(dirPath);
        if (!dir.exists() || !dir.isDirectory()) {
            return new ArrayList<>();
        }
        
        String dotExtension = "." + extension;
        File[] files = dir.listFiles((d, name) -> name.endsWith(dotExtension));
        
        if (files == null) {
            return new ArrayList<>();
        }
        
        return Arrays.stream(files)
                .map(File::getAbsolutePath)
                .collect(Collectors.toList());
    }
    
    /**
     * Calculate entropy of a collection of values
     * 
     * @param values Collection of values
     * @return Entropy value
     */
    public static double calculateEntropy(Collection<Double> values) {
        if (values == null || values.isEmpty()) {
            return 0.0;
        }
        
        // Count frequency of each value
        Map<Double, Integer> frequencyMap = new HashMap<>();
        values.forEach(value -> frequencyMap.put(value, frequencyMap.getOrDefault(value, 0) + 1));
        
        // Calculate entropy
        double entropy = 0.0;
        int total = values.size();
        
        for (int frequency : frequencyMap.values()) {
            double probability = (double) frequency / total;
            entropy -= probability * Math.log(probability) / Math.log(2);
        }
        
        return entropy;
    }
    
    /**
     * Calculate the mean absolute percentage error (MAPE) between predicted and actual values
     * 
     * @param predicted Array of predicted values
     * @param actual Array of actual values
     * @return MAPE value, or Double.MAX_VALUE if arrays have different lengths
     */
    public static double calculateMAPE(double[] predicted, double[] actual) {
        if (predicted == null || actual == null || predicted.length != actual.length) {
            return Double.MAX_VALUE;
        }
        
        double sumAbsolutePercentageError = 0.0;
        int validCount = 0;
        
        for (int i = 0; i < predicted.length; i++) {
            if (actual[i] != 0) {
                sumAbsolutePercentageError += Math.abs((actual[i] - predicted[i]) / actual[i]);
                validCount++;
            }
        }
        
        if (validCount == 0) {
            return Double.MAX_VALUE;
        }
        
        return (sumAbsolutePercentageError / validCount) * 100;
    }
    
    /**
     * Calculate exponential moving average
     * 
     * @param currentValue Current value
     * @param previousEMA Previous EMA value
     * @param alpha Smoothing factor (0 < alpha <= 1)
     * @return New EMA value
     */
    public static double calculateEMA(double currentValue, double previousEMA, double alpha) {
        if (alpha <= 0 || alpha > 1) {
            throw new IllegalArgumentException("Alpha must be between 0 and 1 (exclusive-inclusive)");
        }
        
        return alpha * currentValue + (1 - alpha) * previousEMA;
    }
    
    /**
     * Create a thread-safe cache with limited capacity using LRU eviction policy
     * 
     * @param <K> Key type
     * @param <V> Value type
     * @param capacity Maximum number of entries in the cache
     * @return Thread-safe cache map
     */
    public static <K, V> Map<K, V> createLRUCache(final int capacity) {
        return new ConcurrentHashMap<K, V>() {
            private static final long serialVersionUID = 1L;
            private final List<K> accessOrder = new ArrayList<>();
            
            @Override
            public V put(K key, V value) {
                synchronized (accessOrder) {
                    accessOrder.remove(key);
                    accessOrder.add(key);
                    
                    if (accessOrder.size() > capacity) {
                        K oldest = accessOrder.remove(0);
                        super.remove(oldest);
                    }
                }
                
                return super.put(key, value);
            }
            
            @Override
            public V get(Object key) {
                V value = super.get(key);
                
                if (value != null) {
                    synchronized (accessOrder) {
                        accessOrder.remove(key);
                        accessOrder.add((K) key);
                    }
                }
                
                return value;
            }
            
            @Override
            public V remove(Object key) {
                synchronized (accessOrder) {
                    accessOrder.remove(key);
                }
                
                return super.remove(key);
            }
        };
    }
}