import pandas as pd

df = pd.read_csv("data/Indian_Food_Nutrition_Processed.csv")

goitrogenic = ["cabbage", "cauliflower", "spinach", "broccoli", "mustard", "soy", "tofu", "peanut", "radish"]
thyroid_supportive = ["fish", "egg", "milk", "yogurt", "curd", "cheese", "iodized salt", "brazil nut", "almond", "cashew"]

def tag_thyroid_impact(food):
    f = str(food).lower()
    if any(g in f for g in goitrogenic):
        return "Goitrogenic – Limit in Hypothyroidism"
    elif any(s in f for s in thyroid_supportive):
        return "Thyroid Supportive – Good for Thyroid Health"
    else:
        return "Neutral – No major thyroid impact"

df["Thyroid_Impact"] = df["Dish Name"].apply(tag_thyroid_impact)

df.to_csv("data/Indian_Food_Nutrition_Processed.csv", index=False)
print("Thyroid_Impact column added successfully!")
