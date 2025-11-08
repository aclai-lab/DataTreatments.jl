using Test
using DataTreatments

X = rand(100)
wfunc = splitwindow(nwindows=10)
intervals = @evalwindow X wfunc
result = applyfeat(X, intervals)

X = rand(100, 120)
wfunc = splitwindow(nwindows=3)
intervals = @evalwindow X wfunc
result = applyfeat(X, intervals; reducefunc=maximum)

X = rand(100, 120)
Xmatrix = fill(X, 100, 10)
wfunc = splitwindow(nwindows=3)
intervals = @evalwindow X wfunc
result = reducesize(Xmatrix, intervals; reducefunc=std)

X = rand(100, 120)
Xmatrix = fill(X, 100, 10)
wfunc = splitwindow(nwindows=3)
intervals = @evalwindow X wfunc
features = (mean, maximum)
result = aggregate(Xmatrix, intervals; features)

# Single window feature
fid = FeatureId(:temperature, mean, 1)
# Displays as: mean(temperature)

# Multi-window feature
fid = FeatureId(:pressure, maximum, 3)
# Displays as: maximum(pressure)_w3

# Access metadata
get_vname(fid)    # :pressure
get_feature(fid)  # maximum
get_nwin(fid)     # 3

Xmatrix = fill(rand(200, 120), 100, 10)  # 100 samples, 10 variables
win = splitwindow(nwindows=4)
features = (mean, std, maximum)

dt = DataTreatment(Xmatrix, :reducesize; 
                   vnames=Symbol.("var", 1:10),
                   win=(win,), 
                   features=features)
# Each 200×120 element becomes 4×4, resulting in 100×10 output

dt = DataTreatment(Xmatrix, :aggregate;
                   vnames=Symbol.("var", 1:10),
                   win=(win,),
                   features=features)
# Returns 100×(10×3×16) = 100×480 flat matrix
# 10 vars × 3 features × 16 windows (4×4 grid)

using DataFrames

df = DataFrame(
    channel1 = [rand(200, 120) for _ in 1:1000],
    channel2 = [rand(200, 120) for _ in 1:1000],
    channel3 = [rand(200, 120) for _ in 1:1000]
)

# Define processing parameters
win = adaptivewindow(nwindows=6, overlap=0.15)
features = (mean, std, maximum, minimum, median)

# Process to tabular format
dt = DataTreatment(df, :reducesize; 
                   win=(win,), 
                   features=features)

# Access processed data
X_flat = get_dataset(dt)        # Flat feature matrix
feature_ids = get_featureid(dt) # Feature metadata

# Get all feature metadata
feature_ids = get_featureid(dt)

# Select specific features
mean_features = findall(fid -> get_feature(fid) == mean, feature_ids)
X_means = dt.dataset[:, mean_features]

# Select features from specific variable
ch1_features = findall(fid -> get_vname(fid) == :channel1, feature_ids)
X_ch1 = dt.dataset[:, ch1_features]

# Select features from specific windows
early_windows = findall(fid -> get_nwin(fid) <= 3, feature_ids)
X_early = dt.dataset[:, early_windows]

# All parameters are stored for experiment reproduction
dt = DataTreatment(df, :reducesize; win=(win,), features=features)

# Extract processing metadata
aggrtype = get_aggrtype(dt)       # :reducesize
reduction = get_reducefunc(dt)    # mean
var_names = get_vnames(dt)        # [:channel1, :channel2, :channel3]
feat_funcs = get_features(dt)     # (mean, std, maximum, minimum, median)
n_windows = get_nwindows(dt)      # 6

# Document experiment
println("Processing: $aggrtype mode")
println("Variables: $(join(var_names, ", "))")
println("Features: $(join(nameof.(feat_funcs), ", "))")
println("Windows: $n_windows per dimension")

using DataTreatments

# Create a dataset with multidimensional elements
X = rand(200, 120)  # Example: 200×120 matrix (e.g., spectrogram)
Xmatrix = fill(X, 100, 10)  # 100×10 dataset where each element is a 200×120 matrix

# Define windowing strategy
win = splitwindow(nwindows=4)  # Split into 4 equal windows per dimension

# Compute intervals for the first element
intervals = @evalwindow X win

# Apply multiple statistical features to each window
features = (mean, std, maximum, minimum)
result = aggregate(Xmatrix, intervals; features)
# Result is a 100×640 tabular matrix where each element is flatted

win = splitwindow(nwindows=3)
# Divides data into 3 equal, non-overlapping segments
win = movingwindow(winsize=50, winstep=25)
# Creates overlapping windows of size 50, advancing by 25 points
win = adaptivewindow(nwindows=5, overlap=0.2)
# Creates 5 windows with 20% overlap between consecutive windows
win = wholewindow()
# Creates a single window covering the entire dimension


X = rand(200, 120)

# Apply same windowing to all dimensions
intervals = @evalwindow X splitwindow(nwindows=4)

# Apply different windowing per dimension
intervals = @evalwindow X splitwindow(nwindows=4) movingwindow(winsize=40, winstep=20)

X = rand(200, 120)
intervals = @evalwindow X splitwindow(nwindows=4)

# Apply mean to each window
result = applyfeat(X, intervals; reducefunc=mean)
# Returns a 4×4 matrix (4 windows per dimension)

Xmatrix = fill(rand(200, 120), 100, 10)  # Dataset of matrices
intervals = @evalwindow first(Xmatrix) splitwindow(nwindows=3)

# Aggregate each element using reduce feature
result = reducesize(Xmatrix, intervals; reducefunc=mean)
# Each element reduced from 200×120 to a 3×3 matrix per feature

Xmatrix = fill(rand(200, 120), 100, 10)  # 100 samples, 10 variables
intervals = @evalwindow first(Xmatrix) splitwindow(nwindows=4)
features = (mean, std, maximum, minimum)

result = aggregate(Xmatrix, intervals; features)
# Returns 100×640 matrix (10 vars × 4 features × 16 windows)

# Created automatically by DataTreatment
dt = DataTreatment(df, :reducesize; win=(win,), features=(mean, std, maximum))

# Access feature metadata
feature_ids = get_featureid(dt)

# Each FeatureId contains:
# - vname: Source variable name
# - feat: Feature function applied
# - nwin: Window number

# Use for feature selection
mean_features = filter(fid -> get_feature(fid) == mean, feature_ids)
temp_features = filter(fid -> get_vname(fid) == :temperature, feature_ids)
window1_features = filter(fid -> get_nwin(fid) == 1, feature_ids)

using DataFrames

# Create dataset
df = DataFrame(
    channel1 = [rand(200, 120) for _ in 1:1000],
    channel2 = [rand(200, 120) for _ in 1:1000],
    channel3 = [rand(200, 120) for _ in 1:1000]
)

# Process with full parameter storage
win = adaptivewindow(nwindows=6, overlap=0.15)
features = (mean, std, maximum, minimum, median)

dt = DataTreatment(df, :reducesize; 
                   win=(win,), 
                   features=features)

# Access processed data
X_flat = get_dataset(dt)        # Flat feature matrix
feature_ids = get_featureid(dt) # Feature metadata

# All parameters are stored for reproducibility
aggrtype = get_aggrtype(dt)     # :reducesize
reduction = get_reducefunc(dt)   # mean (default)
var_names = get_vnames(dt)       # [:channel1, :channel2, :channel3]
feat_funcs = get_features(dt)    # (mean, std, maximum, minimum, median)
n_windows = get_nwindows(dt)     # 6

# Document experiment
println("Processing: $aggrtype mode")
println("Variables: $(join(var_names, ", "))")
println("Features: $(join(nameof.(feat_funcs), ", "))")
println("Windows: $n_windows per dimension")