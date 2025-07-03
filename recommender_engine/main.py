import pandas as pd

from sqlalchemy import create_engine, text, Engine

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

# Set seed for reproducable results
SEED = 1

# Define product columns
product_cols = [
    "investment_account",
    "disability_insurance",
    "retail",
    "accounting_link",
    "company_insurance",
    "accounting_package",
    "pension",
    "credit_card",
    "travel_insurance",
    "job_insurance",
    "legal_insurance",
    "accident_insurance",
]

def run_query(engine: Engine, query: str):
    """Runs an SQL query on an engine.

    Args:
        engine (Engine): The SQL engine.
        query (str): The SQL query.

    Returns:
        pd.DataFrame: A pandas dataframe containing the query results.
    """
    results = None
    with engine.connect() as conn:
        results = pd.DataFrame(conn.execute(text(query)).fetchall())
    return results

def load_pickle_into_db(input_path, engine) -> pd.DataFrame:
    """Loads the pickle file into a database.

    Args:
        input_path (str): Path to the pickle file.
        engine (Engine): The SQL engine to use for storing the data.

    Returns:
        pd.DataFrame: The dataframe that contains the loaded data.
    """
    # Load pickle file into a dataframe
    print(f"Loading data from pickle file '{input_path}'...")
    df = pd.read_pickle(input_path)

    # Fix column name and age values
    df = df.rename(columns={"travel_insurace": "travel_insurance"})
    df["age"] = df["age"].map(lambda x: float(x) if x != ' NA' else None)

    # Sort by customer_id and date_partition
    df = df.sort_values(by=['customer_id', 'date_partition'])

    # Apply shift to get the previous date_partition on which the same customer was updated
    # We will use that to join the target variable with the features (which require to be 1 month lagged)
    df['next_date_partition'] = df.groupby('customer_id')['date_partition'].shift(-1)

    # Load data into a table
    print("Loading data into a table...")
    df.to_sql(name='customer_product', con=engine, if_exists="replace", index=False, chunksize=10000)
    return df

def generate_purchases_table(engine):
    """Generates the purchases table.

    Args:
        engine (Engine): The SQL engine.
    """
    # For each product we will generate a table that contains all purchases.
    # We will perform this for all reoccurances of such events per customer.
    results_df_purchases = pd.DataFrame()
    results_df_churns = pd.DataFrame()
    for product in product_cols:
        sql = f"""
        SELECT 
            *,
            1.0 - purchased AS churned
        FROM
        (
            SELECT 
                customer_id,
                date_partition,
                product_cd,
                CASE
                    WHEN curr_status > prev_status THEN 1.0
                    ELSE 0.0
                END as purchased
            FROM
            (
                SELECT 
                    customer_id,
                    date_partition,
                    '{product}' as product_cd,
                    LAG({product}, 1, 0) OVER (PARTITION BY customer_id ORDER BY date_partition ASC) as prev_status, 
                    {product} as curr_status
                FROM customer_product
            )
            where prev_status != curr_status
        )
        """
        print(f"Running for '{product}'...", end="")
        result = run_query(engine, sql)
        print(f" [DONE] (n={len(result)})")
        
        # Save results to tables
        results_df_purchases = pd.concat([results_df_purchases, result[result["purchased"]==1.0]], axis=0)
        results_df_churns = pd.concat([results_df_churns, result[result["churned"]==1.0]], axis=0)

    results_df_purchases.to_sql(name='product_purchases', con=engine, if_exists="replace")
    results_df_churns.to_sql(name='product_churns', con=engine, if_exists="replace")

def train_test_models(products, df):
    """Trains a model for each product and evaluates it on a test set.

    Args:
        product (List<str>): List of products to run for.
        df (pd.DataFrame): The dataframe containing the dataset.
    """
    # Define the features and target
    num_features = ["age", "customer_seniority", "gross_income", "times_purchased", "days_since_last_purchase"]
    cat_features = ["sex", "new_customer", "acquisition_channel", "activity_status", "education_segment", "sbi", "date_month"]
    cat_features = cat_features + product_cols
    features = cat_features + num_features
    target = "target"

    products = [
        "retail", "company_insurance", "accounting_package", "pension", "credit_card", 
        "travel_insurance", "job_insurance", "legal_insurance", "accident_insurance"
    ]
    feature_importances = {}

    for product in products:
        sql = f"""
        SELECT
            customer_id,
            date_partition,
            times_purchased,
            julianday(date_partition) - julianday(last_purchase) as days_since_last_purchase,
            strftime('%m', date_partition) as date_month,
            target
        FROM
        (
            SELECT
                customer_id,
                date_partition,
                coalesce(sum(purchased) over (partition by customer_id order by date_partition asc rows between unbounded preceding and 1 preceding),0) as times_purchased,
                purchased as target,
                MAX(purchase_dt) OVER (
                    PARTITION BY customer_id
                    ORDER BY date_partition ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS last_purchase
            FROM
            (
                SELECT 
                    a.*,
                    coalesce(b.purchased, 0) as purchased,
                    purchase_dt
                FROM customer_product a
                LEFT JOIN
                (
                    SELECT
                        customer_id,
                        date_partition as purchase_dt,
                        purchased
                    FROM product_purchases
                    WHERE
                        product_cd = '{product}'
                ) b
                ON a.customer_id = b.customer_id and a.next_date_partition = b.purchase_dt
            )
        )
        """
        print(f"Running for '{product}'...")

        # Get dataset
        print("\tBuilding dataset...")
        dataset = run_query(engine, sql)
        dataset = pd.merge(df, dataset, on=["customer_id", "date_partition"])
        dataset = dataset.drop_duplicates(subset=features+[target])

        # Downsample majority class
        majority_class = dataset[dataset['target'] == 0]
        minority_class = dataset[dataset['target'] == 1]
        majority_class_downsampled = majority_class.sample(n=len(minority_class), random_state=SEED)
        dataset = pd.concat([majority_class_downsampled, minority_class])
        dataset = dataset.sample(frac=1, random_state=SEED).reset_index(drop=True)
        dataset = dataset.sort_values(by=["date_partition", "customer_id"], ascending=True)
        
        # Spit into X,y pairs
        print("\tPreparing X,y...")
        X = dataset[features]
        y = dataset[target]

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=SEED)

        # Create transformers for both numerical and categorical columns
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine transformers using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, num_features),
                ('cat', categorical_transformer, cat_features)
            ]
        )
        
        # Define the full pipeline, including a classifier
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=SEED, class_weight="balanced"))
        ])
        
        # Train model on training set
        print("\tTraining model...")
        model = pipeline.fit(X_train, y_train)

        # Predict on test set
        print("\tEvaluation on test set...")
        y_pred = model.predict_proba(X_test)[:,1]
        print(f"\tF1: {f1_score(y_test, y_pred > 0.5)}")
        print(f"\tPrecision: {precision_score(y_test, y_pred > 0.5)}")
        print(f"\tRecall: {recall_score(y_test, y_pred > 0.5)}")
        print(f"\tAUC: {roc_auc_score(y_test, y_pred)}")

        feature_importances[product] = {key: value for key, value in zip(features, model[-1][1].feature_importances_)}
        feature_importances[product] = dict(sorted(feature_importances[product].items(), key=lambda item: item[1], reverse=True))

if __name__ == "__main__":
    engine = create_engine('sqlite://', echo=False)
    products = [
        "retail", "company_insurance", "accounting_package", "pension", "credit_card", 
        "travel_insurance", "job_insurance", "legal_insurance", "accident_insurance"
    ]
    df = load_pickle_into_db("customer_product.pkl", engine)
    generate_purchases_table(engine)
    train_test_models(products, df)
