using MLJ, Imbalance

SMOTE = @load SMOTE pkg=Imbalance verbosity=0
TomekUndersampler = @load TomekUndersampler pkg=Imbalance verbosity=0
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0

oversampler = SMOTE(k=5, ratios=1.0, rng=42)
undersampler = TomekUndersampler(min_ratios=0.5, rng=42)

model = DecisionTreeClassifier()

balanced_model = BalancedModel(model=model, 
                               balancer1=oversampler, balancer2=undersampler)

num_rows = 500
num_features = 2
# generating continuous features given mean and std
X, y = generate_imbalanced_data(
	num_rows,
	num_features;
	means = [1.0, 4.0, [7.0 9.0]],
	stds = [1.0, [0.5 0.8], 2.0],
	class_probs=[0.5, 0.2, 0.3],
	type="Matrix",
	rng = 42,
)
checkbalance(y)

m = machine(balanced_model, X, y)

Xm = m.args[1].data
ym = m.args[2].data
checkbalance(ym)

x1,y1 = smote(X,y,k=5, ratios=0.75, rng=42)
x2,y2 = smote(X,y,k=5, ratios=0.75, rng=42)
y1==y2
checkbalance(y1)
# test2 = tomek_undersample(min_ratios=0.5, rng=42)

a=load_dataset(X,["1","2"],y; balance=(SMOTE(k=5, ratios=0.75, rng=42)))

function q(i::Int)
    @show i
end