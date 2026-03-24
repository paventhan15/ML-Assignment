import pandas as pd
import numpy as np

# Create a sample Amazon Products dataset
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'ProductName': [f'Product_{i}' for i in range(n)],
    'Category': np.random.choice(['Electronics', 'Books', 'Clothing', 'Home', 'Sports'], n),
    'Price': np.random.uniform(10, 500, n),
    'Rating': np.random.uniform(1, 5, n),
    'NumReviews': np.random.randint(0, 1000, n),
    'Discount': np.random.uniform(0, 50, n),
    'InStock': np.random.choice(['Yes', 'No'], n),
    'Description': [f'Sample product description {i}' for i in range(n)]
})

df.to_csv('Amazon-Products.csv', index=False)
print('✅ Sample dataset created: Amazon-Products.csv')
print(f'Shape: {df.shape}')
print(df.head())
