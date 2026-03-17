import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = [
    ("Where is my order?", "Order Status"),
    ("My delivery is late", "Delivery Delay"),
    ("I want a refund", "Refund Request"),
    ("Product is damaged", "Product Issue"),
    ("Wrong item received", "Product Issue"),
    ("Still not delivered", "Delivery Delay"),
    ("Track my order", "Order Status"),
    ("Return my product", "Refund Request"),
    ("Item not working", "Product Issue"),
    ("Order not shipped yet", "Order Status"),
]

df = pd.DataFrame(data, columns=["text", "label"])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

def classify_query(query):
    vec = vectorizer.transform([query])
    return model.predict(vec)[0]

def auto_reply(category):
    responses = {
        "Order Status": "You can track your order here: [link]",
        "Delivery Delay": "Sorry for delay. Your order will arrive soon.",
        "Refund Request": "Refund initiated. It will take 3-5 days.",
        "Product Issue": "We apologize. Please return the item.",
    }
    return responses.get(category, "We will contact you shortly.")

queries = [
    "Where is my order?",
    "Refund please",
    "Late delivery again",
    "Item broken",
    "Track order",
    "Need refund",
]

results = []

print("\n--- CUSTOMER SUPPORT RESPONSES ---\n")

for q in queries:
    category = classify_query(q)
    reply = auto_reply(category)
    
    results.append(category)
    
    print(f"Query: {q}")
    print(f"Category: {category}")
    print(f"Reply: {reply}")
    print("-" * 40)

df_results = pd.DataFrame({
    "query": queries,
    "category": results
})

distribution = df_results["category"].value_counts(normalize=True) * 100

print("\n--- ISSUE DISTRIBUTION (%) ---\n")
print(distribution)

distribution.plot(kind="bar")
plt.title("Customer Issue Distribution")
plt.xlabel("Category")
plt.ylabel("Percentage (%)")
plt.show()

print("\n--- WORKFLOW ---\n")
print("""
User Message
   ↓
AI Model (Classification)
   ↓
Category Output
   ↓
Auto Reply
   ↓
Dashboard Update
""")