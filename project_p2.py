import streamlit as st
import pandas as pd
import joblib

# Load the saved model (pipeline)
model = joblib.load("best_model.joblib")

# Load Pokémon stats
pokemon_stats = pd.read_csv("pokemon.csv")
pokemon_stats['Type2'] = pokemon_stats['Type2'].fillna("None")

st.set_page_config(
    page_title="Pokésight",
    page_icon=":red_circle:",
)
st.title('Pokésight')
st.write("### Pokémon Battle Predictor")
st.write("Enter the names of two Pokémon to predict the outcome of a battle.")

# Input fields for Pokémon names (updates trigger re-run)
p1_name_input = st.text_input("Enter the name of the first Pokémon", value="Bulbasaur", key="p1")
p2_name_input = st.text_input("Enter the name of the second Pokémon", value="Squirtle", key="p2")

# Update and show images based on current text input
p1_candidates = pokemon_stats[pokemon_stats["Name"].str.lower() == p1_name_input.strip().lower()]
p2_candidates = pokemon_stats[pokemon_stats["Name"].str.lower() == p2_name_input.strip().lower()]

if not p1_candidates.empty and not p2_candidates.empty:
    # Save original names for display
    p1_display = p1_candidates["Name"].iloc[0]
    p2_display = p2_candidates["Name"].iloc[0]
    
    # Construct image URLs (replace spaces with hyphens, etc.)
    p1_url = f"https://img.pokemondb.net/artwork/avif/{p1_display.lower().replace(' ', '-')}.avif"
    p2_url = f"https://img.pokemondb.net/artwork/avif/{p2_display.lower().replace(' ', '-')}.avif"
    
    # Display the images side by side using Streamlit columns
    col1, col2, col3 = st.columns(3)
    col1.image(p1_url, caption=p1_display, width=200)
    with col2:
        st.markdown(
            """
            <div style="display: flex; 
                        flex-direction: column; 
                        justify-content: center; 
                        align-items: center; 
                        height: 250px;">
                <h2 style="margin: auto; text-align: center;">VS</h2>
            </div>
            """, unsafe_allow_html=True)
    col3.image(p2_url, caption=p2_display, width=200)

# Prediction section: Only runs when the button is clicked
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Predict Outcome"):
        # Match Pokémon by name (case-insensitive) again to get the proper rows
        p1 = pokemon_stats[pokemon_stats["Name"].str.lower() == p1_name_input.strip().lower()]
        p2 = pokemon_stats[pokemon_stats["Name"].str.lower() == p2_name_input.strip().lower()]
        
        if p1.empty or p2.empty:
            st.error("One or both Pokémon names not found. Please check your spelling.")
        else:
            # Save original names for display
            p1_display = p1["Name"].iloc[0]
            p2_display = p2["Name"].iloc[0]
            
            # Prefix the columns to match the training data structure
            p1 = p1.add_prefix("p1_")
            p2 = p2.add_prefix("p2_")
            
            # Combine the Pokémon data into one row (for the matchup)
            matchup = pd.concat([p1.reset_index(drop=True), p2.reset_index(drop=True)], axis=1)
            matchup['p1_goes_first'] = 1  # Assuming the first Pokémon goes first
            
            # Drop identifiers not used as features
            matchup_features = matchup.drop(columns=["p1_#", "p1_Name", "p2_#", "p2_Name"], errors="ignore")
            
            # Make a prediction using the pipeline (the pipeline does preprocessing internally)
            prediction = model.predict(matchup_features)
            prediction_proba = model.predict_proba(matchup_features)
            
            # Create a dictionary mapping probabilities
            proba_dict = {
                "First Pokémon wins": prediction_proba[0][1],
                "Second Pokémon wins": prediction_proba[0][0]
            }
            
            # Interpret prediction: 1 means first Pokémon wins, 0 means second wins.
            if prediction[0] == 1:
                winner_text = f"{p1_display} wins!"
            else:
                winner_text = f"{p2_display} wins!"
            
            st.write("# Winner:", winner_text)
            st.write("")
            st.write("")
            # Display prediction probabilities using columns with custom spacing
            st.write("Prediction Probabilities:")
            col1, col2 = st.columns([0.4, 1])
            with col1:
                st.write(f"- {p1_display} Wins:")
            with col2:
                st.write(f"{(proba_dict['First Pokémon wins']*100):.2f}%")
            
            col3, col4 = st.columns([0.4, 1])
            with col3:
                st.write(f"- {p2_display} Wins:")
            with col4:
                st.write(f"{(proba_dict['Second Pokémon wins']*100):.2f}%")
            
            st.write("")
            st.write("")
            st.write("Feature Importances:")
            st.image("./PokeSightFeatureImportances.png", width=600)
