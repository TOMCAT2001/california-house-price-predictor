{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "fb882878-5622-4991-9479-e9db0d8c066d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-10-20 18:01:57.821 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\_mdsol\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2026-10-20 18:01:58.770 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "import joblib\n",
    "import pandas as pd\n",
    "import streamlit as st\n",
    "\n",
    "st.set_page_config(page_title=\"House Price Predictor\", page_icon=\"🏠\")\n",
    "\n",
    "\n",
    "@st.cache_resource\n",
    "def load_model():\n",
    "    return joblib.load(\"house_price_model.joblib\")\n",
    "\n",
    "\n",
    "model = load_model()\n",
    "\n",
    "st.title(\"🏠 House Price Predictor\")\n",
    "st.caption(\"California Housing dataset (neighborhood level) - Gradient Boosting model\")\n",
    "\n",
    "c1, c2 = st.columns(2)\n",
    "med_inc = c1.number_input(\"MedInc (median income, x $10k)\", 0.5, 15.0, 3.5)\n",
    "house_age = c2.number_input(\"HouseAge (years)\", 1.0, 52.0, 29.0)\n",
    "ave_rooms = c1.number_input(\"AveRooms\", 1.0, 20.0, 5.0)\n",
    "ave_bedrms = c2.number_input(\"AveBedrms\", 0.5, 5.0, 1.0)\n",
    "population = c1.number_input(\"Population\", 3.0, 35000.0, 1200.0)\n",
    "ave_occup = c2.number_input(\"AveOccup\", 0.5, 20.0, 3.0)\n",
    "latitude = c1.number_input(\"Latitude\", 32.5, 42.0, 34.2)\n",
    "longitude = c2.number_input(\"Longitude\", -124.5, -114.0, -118.5)\n",
    "\n",
    "if st.button(\"Predict\", type=\"primary\"):\n",
    "    columns = [\n",
    "        \"MedInc\", \"HouseAge\", \"AveRooms\", \"AveBedrms\",\n",
    "        \"Population\", \"AveOccup\", \"Latitude\", \"Longitude\",\n",
    "    ]\n",
    "    row = [[med_inc, house_age, ave_rooms, ave_bedrms,\n",
    "            population, ave_occup, latitude, longitude]]\n",
    "    X = pd.DataFrame(row, columns=columns)\n",
    "\n",
    "    price = model.predict(X)[0] * 100_000\n",
    "    st.success(f\"Estimated median house value: ${price:,.0f}\")\n",
    "    st.caption(\"Typical error of the model is about $42,000. \"\n",
    "               \"Not reliable for prices above $500,000.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "81870ac6-8098-4eef-bb0a-03d036816678",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
