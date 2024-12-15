#!/bin/python3
from sklearn.ensemble import GradientBoostingClassifier

from collections import Counter
import seaborn as sns
import os
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, adjusted_rand_score
import numpy as np

def custom_clustering_score(labels_true, labels_pred):
    """
    Compares two clustering label arrays and returns a custom score.
    The score is based on how many clusters are correctly predicted, considering both
    the number of clusters and whether the samples in the clusters match.
    
    :param labels_true: Ground truth cluster labels.
    :param labels_pred: Predicted cluster labels.
    :return: Custom clustering score.
    """
    # Find the unique clusters in the ground truth and predictions
    unique_clusters_true = np.unique(labels_true)
    unique_clusters_pred = np.unique(labels_pred)
    
    total_clusters_true = len(unique_clusters_true)
    total_clusters_pred = len(unique_clusters_pred)
        
    # Count how many clusters are correctly predicted
    correct_clusters = 0
    
    # Check each true cluster to see if there is a matching cluster in the predicted labels
    for true_cluster in unique_clusters_true:
        # Get indices of samples in the current true cluster
        true_indices = np.where(labels_true == true_cluster)[0]
        
        # Check if there is a predicted cluster that matches exactly
        match_found = False
        for pred_cluster in unique_clusters_pred:
            # Get indices of samples in the predicted cluster
            pred_indices = np.where(labels_pred == pred_cluster)[0]
            
            # If the samples in the true cluster match exactly with those in the predicted cluster
            if np.array_equal(np.sort(true_indices), np.sort(pred_indices)):
                match_found = True
                break
        
        # If a matching cluster was found, count it as correct
        if match_found:
            correct_clusters += 1
    
    # The final score is the proportion of correctly matched clusters
    total_clusters = max(total_clusters_true, total_clusters_pred)
    score = correct_clusters / total_clusters
    
    return score

import numpy as np

def parse_and_process_genotype_data(file_path):
    stop_counter = 0
    print("Parsing the file...")
    data = []
    line_count = 0
    # Reading from the file
    with open(file_path, 'r') as file:
        for line in file:
            # Skip empty lines
            if not line.strip():
                continue
            # Split the line by tabs
            line_data = line.strip().split('\t')
            # Ensure there are at least two columns
            if len(line_data) < 3:
                continue
            # Exclude the first two columns (chromosome and position)
            genotype_data = line_data[2:]

            # Convert the genotype data to integers
            try:
                mapped_data = [int(genotype) for genotype in genotype_data]
            except ValueError as e:
                print(f"Error converting genotype data to integers on line {line_count+1}: {e}")
                continue

            data.append(mapped_data)
            # Increment the line count and print progress for every 1 million lines
            line_count += 1
            if line_count % 1_000_000 == 0:
                stop_counter += 1
                print(f"Parsed {line_count} lines...")
                #if stop_counter == 200:
                #    break
    print("File parsing completed.")

    # Convert the list of lists into a numpy array for better performance
    data_array = np.array(data)
    initial_shape = data_array.shape
    print(f"Initial shape of the data: {initial_shape}")

    # Filter out rows with 0 variance (i.e., rows with only one unique value)
    variances = np.var(data_array, axis=1)
    non_zero_variance_mask = variances > 0
    filtered_data_array = data_array[non_zero_variance_mask, :]
    filtered_shape = filtered_data_array.shape

    print(f"Shape of the data after filtering: {filtered_shape}")

    return initial_shape, filtered_shape, filtered_data_array


file_path = '/home/kousis/work/ddrad/output.txt' 
#file_path = '/home/kousis/work/krasia/data/VAR_Only_bcf/TEST4/snpEff/gtg.txt'
initial_shape, filtered_shape, snp_data = parse_and_process_genotype_data(file_path)

with open('/home/kousis/work/ddrad/samples.txt', "r") as names:
    samples= names.readlines()
with open('/home/kousis/work/ddrad/names.txt', "r") as names:
    samples= names.readlines()
print(len(samples))
def form_clusters(samples):
    varieties = []
    for sample in samples:
        num_of_underscores = sample.count("_")
        variety_name = sample.split("_")
        variety_string = "_".join(variety_name[:num_of_underscores])
        
        print(f"Variety name: {variety_string}")
        varieties.append(variety_string)
    
    return varieties
def form_clusters(samples):
    varieties = []
    for sample in samples:
        variety_name = sample.split("_", 1)  # Split only at the first underscore
        variety_string = variety_name[0]  # Take everything before the first underscore

        print(f"Variety name: {variety_string}")
        varieties.append(variety_string)

    return varieties

varieties=form_clusters(samples)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(varieties)

number_of_clusters=len(list(set(encoded_labels)))
transposed_snp_data = snp_data.T
print(transposed_snp_data.shape)
linked_init = linkage(transposed_snp_data, method='ward')
np.savetxt("genes_X.csv", snp_data, delimiter=",", fmt='%d')
max_clusters = number_of_clusters
init_clusters = fcluster(linked_init, max_clusters, criterion='maxclust')
init_ari_score = adjusted_rand_score(encoded_labels, init_clusters)
custom_score = custom_clustering_score(encoded_labels, init_clusters)      
print(f"Initial ARI Score {init_ari_score}")
print(f"initial CCS: {custom_score}")

# Extract distances from the linkage matrix
distances = linked_init[:, 2]

# Sort the distances to determine the threshold for coloring clusters
sorted_distances = sorted(distances)
threshold = sorted_distances[-max_clusters + 1]

# Print the initial ARI score
init_ari_score = adjusted_rand_score(encoded_labels, init_clusters)
print(f"Initial ARI Score: {init_ari_score}")

# Plotting the dendrogram
plt.figure(figsize=(40, 12))
dendrogram(linked_init, 
           labels=varieties, 
           leaf_rotation=90, 
           leaf_font_size=8, 
           show_leaf_counts=True, 
           color_threshold=threshold)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.margins(0.02)
plt.tight_layout()

# Save the dendrogram as a PDF file
plt.savefig(f"init_de.pdf", dpi=300)

print("Combination Plot: Ready")
print("Script Ended: Thank you for using")
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score

# Assuming you have transposed_snp_data, encoded_labels, max_clusters, and custom_clustering_score function defined
ward_ari_scores = []
average_ari_scores = []
ccs_scores = []
variance_thresholds = []

variances = np.var(transposed_snp_data, axis=0)  # Calculate variance across columns (SNP Positions)

# Loop through thresholds from 0.1 to 1.0
for i in range(1, 11):
    var_threshold = i / 10
    variance_mask = variances > var_threshold
    filtered_data_array = transposed_snp_data[:, variance_mask]  # Apply variance mask
    filtered_shape = filtered_data_array.shape

    print(f"For variance threshold {var_threshold}, array shape is: {filtered_shape}")

    # Perform hierarchical clustering using Ward linkage
    linked_ward = linkage(filtered_data_array, method='ward')
    ward_clusters = fcluster(linked_ward, max_clusters, criterion='maxclust')

    # Perform hierarchical clustering using Average linkage
    linked_average = linkage(filtered_data_array, method='average')
    average_clusters = fcluster(linked_average, max_clusters, criterion='maxclust')

    # Calculate ARI scores for Ward and Average linkage
    ward_ari_score = adjusted_rand_score(encoded_labels, ward_clusters)
    average_ari_score = adjusted_rand_score(encoded_labels, average_clusters)

    # Calculate custom clustering score (assumed to be a function you have)
    custom_score = custom_clustering_score(encoded_labels, ward_clusters)

    print(f"For variance threshold {var_threshold}, Ward ARI is: {ward_ari_score}")
    print(f"For variance threshold {var_threshold}, Average ARI is: {average_ari_score}")
    print(f"For variance threshold {var_threshold}, Custom Score is: {custom_score}")

    # Append the scores for plotting
    variance_thresholds.append(var_threshold)
    ward_ari_scores.append(ward_ari_score)
    average_ari_scores.append(average_ari_score)
    ccs_scores.append(custom_score)

# Now let's plot the results
# Set Seaborn theme
sns.set_theme(style="whitegrid")

# Create a DataFrame for plotting
df = pd.DataFrame({
    'Variance Threshold': variance_thresholds,
    'Ward ARI Score': ward_ari_scores,
    'Average ARI Score': average_ari_scores,
    'Custom Clustering Score': ccs_scores
})

# Convert the DataFrame to a long format for Seaborn's lineplot function
df_melted = pd.melt(df, id_vars='Variance Threshold', value_vars=['Ward ARI Score', 'Average ARI Score', 'Custom Clustering Score'], 
                    var_name='Score Type', value_name='Score')

# Create the line plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_melted, x='Variance Threshold', y='Score', hue='Score Type', marker='o')

# Add titles and labels
plt.title('Clustering Scores vs Variance Threshold', fontsize=16)
plt.xlabel('Variance Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)

# Display the legend
plt.legend(title='Score Type', loc='lower left')

# Display the plot
plt.savefig("dd_variance_threhold.pdf")

variance_mask = variances > 0.7
filtered_data_array = transposed_snp_data[:, variance_mask]  # Apply variance mask correctly
filtered_shape = filtered_data_array.shape

transposed_snp_data=filtered_data_array

# Nested cross-validation for random forest
X=transposed_snp_data
y=encoded_labels
del(transposed_snp_data)
del(snp_data)
new_X=np.unique(X, axis=1)
X=new_X
new_X.shape
# configure the cross-validation procedure
cv_outer = KFold(n_splits=5, shuffle=True, random_state=7)
# enumerate splits
outer_results = list()
outer_clust = list()
counter=0
for train_ix, test_ix in cv_outer.split(new_X):
    counter+=1
    print(counter)
    # split data                                                                                                                                                                                                                                                                
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    # configure the cross-validation procedure                                                                                                                                                                                                                                  
    cv_inner = KFold(n_splits=10, shuffle=True, random_state=1)
    # define the model                                                                                                                                                                                                                                                          
    model = RandomForestClassifier(random_state=1)
    # define search space                                                                                                                                                                                                                                                       
    space = dict()
    space['n_estimators'] = [10, 100, 200, 500]
    # define search                                                                                                                                                                                                                                                             
    search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
    print("HT complete")
    # execute search                                                                                                                                                                                                                                                            
    result = search.fit(X_train, y_train)
    # get the best performing model fit on the whole training set                                                                                                                                                                                                               
    best_model = result.best_estimator_
    feature_importances = best_model.feature_importances_
    important_feature_indices = np.where(feature_importances > 0)[0]

    print("best features done")
    important_X = new_X[:, important_feature_indices]
    np.savetxt(f"train_{counter}", important_X.T, delimiter=" ", fmt='%d')

    linked_var = linkage(important_X, method='ward')
    var_clusters = fcluster(linked_var, max_clusters, criterion='maxclust')
    var_ari_score = adjusted_rand_score(y, var_clusters)
    cc=custom_clustering_score(y, var_clusters)
    print(f"ARI Score after selecting important features (importance > 0): {var_ari_score}")
    acc=var_ari_score
    # store the result                                                                                                                                                                                                                                                          
    print(acc)
    print(cc)
    outer_results.append(acc)
    outer_clust.append(cc)
    # report progress                                                                                                                                                                                                                                                           
    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))

print(np.mean(outer_results))
print(np.mean(outer_clust))
print(outer_clust)



folds= [i for i in range(1,len(outer_clust)+1)]
print(folds)
# Calculate the mean for ARI and CCS scores
mean_ari = np.mean(outer_results)
mean_ccs = np.mean(outer_clust)

# Set the Seaborn theme
sns.set_theme(style="darkgrid")
sns.plotting_context("talk")
# Create a figure and axis
plt.figure(figsize=(10, 6))

# Plot both ARI and CCS scores with lines and markers, using discrete folds
sns.lineplot(x=folds, y=outer_results, marker='o', label='ARI Score')
sns.lineplot(x=folds, y=outer_clust, marker='o', label='Custom Clustering Score (CCS)')

# Add horizontal lines for the mean of ARI and CCS scores
plt.axhline(y=mean_ari, color='blue', linestyle='--', label=f'Mean ARI ({mean_ari:.2f})')
plt.axhline(y=mean_ccs, color='#CC630F', linestyle='--', label=f'Mean CCS ({mean_ccs:.2f})')

# Add titles and labels
plt.title('Comparison of ARI and Custom Clustering Score Across Folds', fontsize=14)
plt.xlabel('Fold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.ylim(0.0, 1.1)
# Set the x-axis ticks to the fold numbers (discrete values)
plt.xticks(ticks=folds)

# Show the legend
plt.legend()
# Show the plot
plt.savefig("dd_accuracy.pdf")















model = RandomForestClassifier(random_state=1)
space = dict()
space['n_estimators'] = [10, 100, 200, 500, 600, 700, 800]

# define search
cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)
search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True,n_jobs=-1)

# execute search
result = search.fit(X, y)
print("HT complete")
# get the best performing model fit on the whole training set
best_model = result.best_estimator_
print(best_model)
feature_importances = best_model.feature_importances_


non_zero_importance_indices = np.where(feature_importances > 0)[0]  # Get indices of non-zero importance features
filtered_importances = feature_importances[non_zero_importance_indices]  # Filter importances
filtered_X = new_X[:, non_zero_importance_indices]  # Filter the features accordingly

print(f"Number of important features: {len(non_zero_importance_indices)}")
print("Important features selection done")
print(f"Filtered X shape: {filtered_X.shape}")

# Step 2: Perform clustering on all important features initially
linked_var = linkage(filtered_X, method='ward')
var_clusters = fcluster(linked_var, max_clusters, criterion='maxclust')
var_ari_score = adjusted_rand_score(y, var_clusters)
print(f"ARI Score after selecting important features (importance > 0): {var_ari_score}")

# Store the initial ARI score
acc = var_ari_score
print(f"Initial ARI Score: {acc}")

# Step 3: Sort the remaining feature importances in descending order
sorted_indices = np.argsort(filtered_importances)[::-1]

# Step 4: Initialize variables for batch processing
batch_size = 100
ari_scores = []
ccs_scores = []
ward_ari_scores = []
# Step 5: Loop through features in batches and perform clustering
for i in range(batch_size, len(sorted_indices) + batch_size, batch_size):
    # Get the top i important features based on sorted indices
    top_indices = sorted_indices[:i]

    # Select these important features from the filtered_X
    important_X_batch = filtered_X[:, top_indices]

    # Perform hierarchical clustering on the important features
    linked_var = linkage(important_X_batch, method='average')  # Using 'average' linkage for clustering
    
    var_clusters = fcluster(linked_var, max_clusters, criterion='maxclust')

    # Compute the ARI score for this batch
    var_ari_score = adjusted_rand_score(y, var_clusters)
    linked_var_ward = linkage(important_X_batch, method='ward')  # Using 'average' linkage for clustering
    ward_var_clusters = fcluster(linked_var_ward, max_clusters, criterion='maxclust')
    ward_var_ari_score = adjusted_rand_score(y, ward_var_clusters)
    # Custom clustering score, assumed to be a function you've defined
    score = custom_clustering_score(y, var_clusters)

    # Print ARI score and custom score after each batch
    print(f"ARI Score after selecting top {i} features: {var_ari_score}")
    print(f"Ward_ARI Score after selecting top {i} features: {ward_var_ari_score}")
    print(f"Custom clustering score: {score}")

    # Append scores to the list for further analysis
    ari_scores.append(var_ari_score)
    ccs_scores.append(score)
    ward_ari_scores.append(ward_var_ari_score)


import pandas as pd
# Set the Seaborn theme
sns.set_theme(style="darkgrid")

# Number of features considered in each batch
num_features = [i for i in range(batch_size, len(ari_scores) * batch_size + 1, batch_size)]

# Create a DataFrame for easy plotting with Seaborn
df = pd.DataFrame({
    'Number of Features': num_features,
    'ARI Score': ari_scores,
    'Ward ARI Score': ward_ari_scores,
    'Custom Clustering Score': ccs_scores
})

# Convert the DataFrame to a long format for Seaborn's lineplot function
df_melted = pd.melt(df, id_vars='Number of Features', value_vars=['ARI Score', 'Ward ARI Score', 'Custom Clustering Score'], 
                    var_name='Score Type', value_name='Score')

# Create the line plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_melted, x='Number of Features', y='Score', hue='Score Type', marker='o')

# Add titles and labels
plt.title('Clustering Performance vs Number of Features', fontsize=16)
plt.xlabel('Number of Features', fontsize=12)
plt.ylabel('Score', fontsize=12)

# Show the legend
plt.legend(title='Score Type', loc='lower right')

# Display the plot
plt.savefig("dd_feature_select.pdf")

important_feature_indices = np.where(feature_importances > 0)[0]
important_X = new_X[:, important_feature_indices]
np.savetxt("dd_all_imp_patterns.csv", important_X.T, delimiter=" ", fmt='%d')
np.savetxt("dd_all_patterns.csv", new_X.T, delimiter=" ", fmt='%d')

top_300_indices = np.argsort(feature_importances)[-50:]
print(f"Number of selected features: {len(top_300_indices)}")
print("best features done")
important_X = new_X[:, top_300_indices]
print(important_X.shape)
# Transpose `important_X` (make columns rows)
important_X_transposed = important_X.T


import matplotlib.pyplot as plt
import numpy as np

# Create a figure and axis with minimalistic styling
plt.clf()

fig, ax = plt.subplots(figsize=(15, 40))
sns.set_theme()
# Define a colormap with distinct colors for 0, 1, and 2
cmap = plt.cm.colors.ListedColormap(['white', 'darkgrey', 'black'])
bounds = [0, 0.5, 1.5, 2.5]
norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

# Transpose the matrix so that rows become columns and vice versa
heatmap = ax.imshow(important_X.T, cmap=cmap, norm=norm, aspect='auto')

# Set labels for y-axis (rows)
ax.set_yticks(np.arange(important_X.shape[1]))

# Set labels for x-axis (columns) with varieties rotated for better readability
ax.set_xticks(np.arange(important_X.shape[0]))
ax.set_xticklabels(varieties)
plt.xticks(rotation=90)

# Add grid lines around each box to create a neat boxed appearance
ax.set_xticks(np.arange(-0.5, important_X.shape[0], 1), minor=True)
ax.set_yticks(np.arange(-0.5, important_X.shape[1], 1), minor=True)
ax.grid(False)
ax.grid(which='minor', color='black', linestyle='-', linewidth=1.2)

# Remove major ticks for a cleaner look
ax.tick_params(which='major', length=0)

# Set the labels for the axes with a clean font
ax.set_ylabel('Columns', fontsize=16)
ax.set_xlabel('Variable Varieties', fontsize=16)

# Use a minimalist theme for the plot by hiding unnecessary spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Save the plot to a file
plt.savefig("dd_p300_minimalistic.pdf", bbox_inches='tight', pad_inches=0.1)
plt.close()
