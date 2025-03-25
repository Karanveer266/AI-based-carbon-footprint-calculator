# Carbon Footprint Calculator Application
# This Streamlit app calculates a user's carbon footprint based on their daily activities
# It uses a step-by-step form to collect data across multiple categories like transportation,
# food consumption, home energy use, and consumer goods
# The app processes this data using the Gemma 3 LLM through OpenRouter's API to provide
# personalized carbon footprint calculations and recommendations

import streamlit as st
import base64
import requests
import json

# Set up the Streamlit page configuration with a wide layout for better form display
st.set_page_config(
    layout="wide",
    page_title="Comprehensive Carbon Footprint Calculator"
)

# API Key for accessing OpenRouter.ai to use the Gemma 3 language model
# This key is used for all AI analysis functions in the application
GEMMA_API_KEY = "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"

def encode_image(image_path: str) -> str:
    """
    Helper function that converts an image file to base64 encoding for API transmission
    Takes a file path and returns the encoded string or None if there's an error
    This encoding is necessary for sending images to the LLM for receipt analysis
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_image
    except Exception as e:
        st.error(f"Error encoding image: {str(e)}")
        return None

def analyze_with_gemma(user_responses):
    """
    Core analysis function that sends user activity data to Gemma 3 LLM for carbon footprint calculation
    Takes the complete dictionary of user responses and returns a detailed markdown analysis
    The function constructs a specialized prompt that guides the AI to calculate emissions
    by category and provide actionable recommendations
    """
    prompt = f"""
    You are an expert in carbon footprint calculation. Based on the following user activities, calculate their daily carbon footprint in kg CO2e.
    Provide detailed breakdown by category and explain the calculation methodology.
    
    User's daily activities:
    {json.dumps(user_responses, indent=2)}
    
    Calculate the carbon footprint with these guidelines:
    1. Use region-specific emission factors where possible (assume global average if no region specified)
    2. Break down the calculation by categories (transportation, food, energy, etc.)
    3. Provide the total carbon footprint in kg CO2e
    4. Include specific recommendations for reducing their carbon footprint
    5. Compare their footprint to global average (which is about 4.5 tons CO2e per year or 12.3 kg CO2e per day)
    6. In the end provide some suggestions to the user on how they can reduce their carbon footprint.
    
    Format your response in markdown with clear headings and sections.
    """
    
    try:
        # Send request to OpenRouter API to access Gemma 3 model
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GEMMA_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://carbon-calculator.app",
                "X-Title": "Carbon Footprint Calculator",
            },
            json={
                "model": "google/gemma-3-4b-it:free",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
            },
            timeout=90
        )
        
        # Handle the API response with comprehensive error checking
        # Different LLM providers may return results in different formats
        if response.status_code == 200:
            try:
                # Parse the JSON response
                response_json = response.json()
                
                # Try to extract content from different possible response formats
                if 'choices' in response_json and len(response_json['choices']) > 0:
                    if 'message' in response_json['choices'][0] and 'content' in response_json['choices'][0]['message']:
                        return response_json['choices'][0]['message']['content']
                    elif 'text' in response_json['choices'][0]:
                        return response_json['choices'][0]['text']
                elif 'response' in response_json:
                    return response_json['response']
                elif 'result' in response_json:
                    return response_json['result']
                elif 'output' in response_json:
                    return response_json['output']
                elif 'text' in response_json:
                    return response_json['text']
                
                # If we can't find the content in any expected location, return the raw JSON as text
                return json.dumps(response_json, indent=2)
            except json.JSONDecodeError:
                # If not valid JSON, return the text directly
                return response.text
        else:
            return f"Error {response.status_code}: The API request failed. Please try again later."
            
    except Exception as e:
        return f"An error occurred: {str(e)}"

def analyze_food_invoice(image_data: str) -> str:
    """
    Specialized function that sends a food receipt/invoice image to Gemma 3 for analysis
    Takes base64-encoded image data and returns a detailed breakdown of the food carbon footprint
    The LLM identifies food items, classifies them, and estimates their environmental impact
    """
    prompt_instruction = """
    Analyze this food order receipt/invoice and provide:
    
    1. A list of all food items, classified as vegetarian or non-vegetarian
    2. An estimate of the carbon footprint for each item using standard emission factors
    3. The total carbon footprint of the order
    4. Suggestions to reduce the carbon footprint of future orders
    
    
    Use India-specific carbon emission factors when available.
    """
    
    try:
        # Send request to OpenRouter API with both text instructions and image data
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GEMMA_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://carbon-calculator.app",
                "X-Title": "Food Carbon Footprint Analyzer",
            },
            json={
                "model": "google/gemma-3-4b-it:free",
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt_instruction},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                    ]}
                ],
            },
            timeout=90
        )
        
        # Process the response with similar error handling as the main analysis function
        if response.status_code == 200:
            try:
                # Parse the JSON response
                response_json = response.json()
                
                # Try to extract content from different possible response formats
                if 'choices' in response_json and len(response_json['choices']) > 0:
                    if 'message' in response_json['choices'][0] and 'content' in response_json['choices'][0]['message']:
                        return response_json['choices'][0]['message']['content']
                    elif 'text' in response_json['choices'][0]:
                        return response_json['choices'][0]['text']
                elif 'response' in response_json:
                    return response_json['response']
                elif 'result' in response_json:
                    return response_json['result']
                elif 'output' in response_json:
                    return response_json['output']
                elif 'text' in response_json:
                    return response_json['text']
                
                # If we can't find the content in any expected location, return the raw JSON as text
                return json.dumps(response_json, indent=2)
            except json.JSONDecodeError:
                # If not valid JSON, return the text directly
                return response.text
        else:
            return f"Error {response.status_code}: The API request failed. Please try again later."
            
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Initialize session state variables to maintain app state between reruns
# These variables track the current step, user responses, and analysis results
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
    
if 'user_responses' not in st.session_state:
    st.session_state.user_responses = {}
    
if 'food_order_analysis' not in st.session_state:
    st.session_state.food_order_analysis = None
    
if 'final_result' not in st.session_state:
    st.session_state.final_result = None

# Helper function to safely get responses with default values
# This prevents errors when accessing data that might not exist yet
def get_response(category, question, default=None):
    if category in st.session_state.user_responses:
        if question in st.session_state.user_responses[category]:
            return st.session_state.user_responses[category][question]
    return default

# Function to store responses in the session state
# Organizes data hierarchically by category and question
def store_response(category, question, response):
    if category not in st.session_state.user_responses:
        st.session_state.user_responses[category] = {}
    st.session_state.user_responses[category][question] = response

# Navigation functions to move between form steps
def next_step():
    st.session_state.current_step += 1
    
def prev_step():
    if st.session_state.current_step > 1:
        st.session_state.current_step -= 1

# Main Streamlit UI header and description
st.title("Comprehensive Carbon Footprint Calculator")
st.markdown("""
This calculator helps you understand your daily carbon footprint based on your activities.
Answer the questions below to get a detailed analysis of your environmental impact.
""")

# Step 1: Transportation - Collects information about how the user travels
if st.session_state.current_step == 1:
    st.header("Transportation")
    
    # Initialize transportation category if needed
    if "transportation" not in st.session_state.user_responses:
        st.session_state.user_responses["transportation"] = {}
    
    # Transportation mode selection with default value handling
    transport_options = ["Car", "Bus", "Train", "Bicycle", "Walking", "Motorcycle", "Airplane", "Other"]
    default_transport = get_response("transportation", "primary_mode", "Car")
    try:
        default_index = transport_options.index(default_transport)
    except ValueError:
        default_index = 0
    
    transport_mode = st.selectbox(
        "What was your primary mode of transportation today?",
        transport_options,
        index=default_index,
        key="transport_mode"
    )
    store_response("transportation", "primary_mode", transport_mode)
    
    # Conditional inputs based on transportation mode
    # Different vehicle types need different information
    if transport_mode in ["Car", "Motorcycle"]:
        fuel_options = ["Petrol/Gasoline", "Diesel", "Electric", "Hybrid", "CNG/LPG"]
        default_fuel = get_response("transportation", "fuel_type", "Petrol/Gasoline")
        try:
            default_fuel_index = fuel_options.index(default_fuel)
        except ValueError:
            default_fuel_index = 0
            
        fuel_type = st.selectbox(
            "What type of fuel does your vehicle use?",
            fuel_options,
            index=default_fuel_index,
            key="fuel_type"
        )
        store_response("transportation", "fuel_type", fuel_type)
    
    # Distance traveled - important for all transport modes
    distance_value = float(get_response("transportation", "distance_km", 0.0))
    distance = st.number_input(
        "How many kilometers did you travel today?", 
        min_value=0.0, 
        step=0.5,
        value=distance_value,
        key="distance"
    )
    store_response("transportation", "distance_km", distance)
    
    # Public transport specific questions
    if transport_mode in ["Bus", "Train"]:
        duration_value = int(get_response("transportation", "duration_minutes", 0))
        duration = st.number_input(
            "How many minutes did you spend on public transport?", 
            min_value=0, 
            step=5,
            value=duration_value,
            key="duration"
        )
        store_response("transportation", "duration_minutes", duration)
    
    # Car occupancy affects per-person emissions
    if transport_mode == "Car":
        passengers_value = int(get_response("transportation", "passengers", 1))
        passengers = st.number_input(
            "Including yourself, how many people were in the vehicle?", 
            min_value=1, 
            step=1,
            value=passengers_value,
            key="passengers"
        )
        store_response("transportation", "passengers", passengers)
    
    # Navigation button to proceed to next step
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("Next: Food & Diet", key="transport_next", use_container_width=True):
            next_step()

# Step 2: Food & Diet - Collects information about the user's eating habits
elif st.session_state.current_step == 2:
    st.header("Food & Diet")
    
    # Initialize food and diet categories
    if "food" not in st.session_state.user_responses:
        st.session_state.user_responses["food"] = {}
    if "diet" not in st.session_state.user_responses:
        st.session_state.user_responses["diet"] = {}
    
    # General diet pattern has a major impact on carbon footprint
    diet_options = ["Omnivore (regular meat consumption)", "Flexitarian (occasional meat)", 
                    "Pescatarian (fish but no meat)", "Vegetarian (no meat or fish)", 
                    "Vegan (no animal products)"]
    default_diet = get_response("diet", "diet_type", "Omnivore (regular meat consumption)")
    try:
        default_diet_index = diet_options.index(default_diet)
    except ValueError:
        default_diet_index = 0
        
    diet_type = st.selectbox(
        "How would you describe your diet?",
        diet_options,
        index=default_diet_index,
        key="diet_type"
    )
    store_response("diet", "diet_type", diet_type)
    
    # Breakfast section
    st.subheader("Breakfast")
    had_breakfast = st.checkbox(
        "Did you have breakfast today?",
        value=get_response("food", "had_breakfast", False),
        key="had_breakfast"
    )
    store_response("food", "had_breakfast", had_breakfast)
    
    # Additional questions if user had breakfast
    if had_breakfast:
        breakfast = st.text_area(
            "Please describe what you ate for breakfast",
            value=get_response("food", "breakfast_description", ""),
            key="breakfast_desc"
        )
        store_response("food", "breakfast_description", breakfast)
        
        breakfast_dairy = st.slider(
            "How much dairy did your breakfast contain?", 
            0, 5, 
            value=int(get_response("food", "breakfast_dairy_level", 0)),
            help="0 = none, 5 = large amounts (e.g., milk, cheese, yogurt)",
            key="breakfast_dairy"
        )
        store_response("food", "breakfast_dairy_level", breakfast_dairy)
        
        breakfast_meat = st.slider(
            "How much meat/eggs did your breakfast contain?", 
            0, 5, 
            value=int(get_response("food", "breakfast_meat_level", 0)),
            help="0 = none, 5 = large amounts",
            key="breakfast_meat"
        )
        store_response("food", "breakfast_meat_level", breakfast_meat)
    
    # Lunch section with source tracking
    st.subheader("Lunch")
    had_lunch = st.checkbox(
        "Did you have lunch today?",
        value=get_response("food", "had_lunch", False),
        key="had_lunch"
    )
    store_response("food", "had_lunch", had_lunch)
    
    # Additional questions if user had lunch
    if had_lunch:
        lunch_source_options = ["Home-cooked", "Restaurant", "Office/School Cafeteria", "Delivery/Takeout", "Other"]
        default_lunch_source = get_response("food", "lunch_source", "Home-cooked")
        try:
            default_lunch_source_index = lunch_source_options.index(default_lunch_source)
        except ValueError:
            default_lunch_source_index = 0
            
        lunch_source = st.radio(
            "Where did you get your lunch?", 
            lunch_source_options,
            index=default_lunch_source_index,
            key="lunch_source"
        )
        store_response("food", "lunch_source", lunch_source)
        
        # Special handling for delivery/takeout with invoice upload option
        if lunch_source == "Delivery/Takeout":
            has_invoice = st.checkbox(
                "Do you have the delivery invoice/receipt?",
                value=get_response("food", "has_lunch_invoice", False),
                key="lunch_invoice"
            )
            store_response("food", "has_lunch_invoice", has_invoice)
            
            if has_invoice:
                st.info("You'll be able to upload the invoice in a later step.")
            else:
                lunch_desc = st.text_area(
                    "Please describe what you ate for lunch",
                    value=get_response("food", "lunch_description", ""),
                    key="lunch_desc"
                )
                store_response("food", "lunch_description", lunch_desc)
                
                lunch_meat = st.slider(
                    "How much meat did your lunch contain?", 
                    0, 5, 
                    value=int(get_response("food", "lunch_meat_level", 0)),
                    help="0 = none, 5 = large amounts",
                    key="lunch_meat"
                )
                store_response("food", "lunch_meat_level", lunch_meat)
        else:
            lunch_desc = st.text_area(
                "Please describe what you ate for lunch",
                value=get_response("food", "lunch_description", ""),
                key="lunch_desc_2"
            )
            store_response("food", "lunch_description", lunch_desc)
            
            lunch_meat = st.slider(
                "How much meat did your lunch contain?", 
                0, 5, 
                value=int(get_response("food", "lunch_meat_level", 0)),
                help="0 = none, 5 = large amounts",
                key="lunch_meat_2"
            )
            store_response("food", "lunch_meat_level", lunch_meat)
    
    # Dinner section with similar structure to lunch
    st.subheader("Dinner")
    had_dinner = st.checkbox(
        "Did you have dinner?",
        value=get_response("food", "had_dinner", False),
        key="had_dinner"
    )
    store_response("food", "had_dinner", had_dinner)
    
    # Additional questions if user had dinner
    if had_dinner:
        dinner_source_options = ["Home-cooked", "Restaurant", "Delivery/Takeout", "Other"]
        default_dinner_source = get_response("food", "dinner_source", "Home-cooked")
        try:
            default_dinner_source_index = dinner_source_options.index(default_dinner_source)
        except ValueError:
            default_dinner_source_index = 0
            
        dinner_source = st.radio(
            "Where did you get your dinner?", 
            dinner_source_options,
            index=default_dinner_source_index,
            key="dinner_source"
        )
        store_response("food", "dinner_source", dinner_source)
        
        # Similar invoice option for dinner delivery
        if dinner_source == "Delivery/Takeout":
            has_dinner_invoice = st.checkbox(
                "Do you have the dinner delivery invoice/receipt?",
                value=get_response("food", "has_dinner_invoice", False),
                key="dinner_invoice"
            )
            store_response("food", "has_dinner_invoice", has_dinner_invoice)
            
            if has_dinner_invoice:
                st.info("You'll be able to upload the invoice in a later step.")
            else:
                dinner_desc = st.text_area(
                    "Please describe what you ate for dinner",
                    value=get_response("food", "dinner_description", ""),
                    key="dinner_desc"
                )
                store_response("food", "dinner_description", dinner_desc)
                
                dinner_meat = st.slider(
                    "How much meat did your dinner contain?", 
                    0, 5, 
                    value=int(get_response("food", "dinner_meat_level", 0)),
                    help="0 = none, 5 = large amounts",
                    key="dinner_meat"
                )
                store_response("food", "dinner_meat_level", dinner_meat)
        else:
            dinner_desc = st.text_area(
                "Please describe what you ate for dinner",
                value=get_response("food", "dinner_description", ""),
                key="dinner_desc_2"
            )
            store_response("food", "dinner_description", dinner_desc)
            
            dinner_meat = st.slider(
                "How much meat did your dinner contain?", 
                0, 5, 
                value=int(get_response("food", "dinner_meat_level", 0)),
                help="0 = none, 5 = large amounts",
                key="dinner_meat_2"
            )
            store_response("food", "dinner_meat_level", dinner_meat)
    
    # Additional food consumption like snacks and food waste
    st.subheader("Other Food Consumption")
    snacks = st.text_area(
        "Did you have any snacks or beverages today? Please describe",
        value=get_response("food", "snacks_description", ""),
        key="snacks"
    )
    store_response("food", "snacks_description", snacks)
    
    # Food waste has significant impact on overall footprint
    food_waste = st.slider(
        "How much food did you throw away today?", 
        0, 5, 
        value=int(get_response("food", "food_waste_level", 0)),
        help="0 = none, 5 = significant amount",
        key="food_waste"
    )
    store_response("food", "food_waste_level", food_waste)
    
    # Navigation buttons for this step
    col1, col2, col3 = st.columns([1, 1, 5])
    with col1:
        if st.button("Previous", key="food_prev", use_container_width=True):
            prev_step()
    with col2:
        if st.button("Next: Home Energy", key="food_next", use_container_width=True):
            next_step()

# Step 3: Home Energy - Collects information about household energy and water usage
elif st.session_state.current_step == 3:
    st.header("Home Energy")
    
    # Initialize needed categories for this section
    for category in ["home", "energy", "water"]:
        if category not in st.session_state.user_responses:
            st.session_state.user_responses[category] = {}
    
    # Home type and size information
    home_type_options = ["Apartment", "Small house", "Medium house", "Large house", "Other"]
    default_home_type = get_response("home", "home_type", "Apartment")
    try:
        default_home_type_index = home_type_options.index(default_home_type)
    except ValueError:
        default_home_type_index = 0
    
    home_type = st.selectbox(
        "What type of home do you live in?",
        home_type_options,
        index=default_home_type_index,
        key="home_type"
    )
    store_response("home", "home_type", home_type)
    
    # Household size affects per-person footprint
    household_size = st.number_input(
        "How many people live in your household?", 
        min_value=1, 
        step=1,
        value=int(get_response("home", "household_size", 1)),
        key="household_size"
    )
    store_response("home", "household_size", household_size)
    
    # Electricity sources and usage
    st.subheader("Electricity")
    electricity_options = ["Grid electricity", "Solar panels", "Wind power", "Other renewable", "Don't know"]
    default_electricity = get_response("energy", "electricity_sources", [])
    
    electricity_source = st.multiselect(
        "What are your sources of electricity? (Select all that apply)",
        electricity_options,
        default=default_electricity,
        key="electricity_source"
    )
    store_response("energy", "electricity_sources", electricity_source)
    
    # Conditional question for grid electricity users
    if "Grid electricity" in electricity_source:
        electricity_provider = st.text_input(
            "Who is your electricity provider? (Optional)",
            value=get_response("energy", "electricity_provider", ""),
            key="electricity_provider"
        )
        store_response("energy", "electricity_provider", electricity_provider)
    
    # Heating and cooling have major energy impacts
    st.subheader("Heating & Cooling")
    ac_usage = st.slider(
        "How many hours did you use air conditioning today?", 
        0, 24, 
        value=int(get_response("energy", "ac_hours", 0)),
        key="ac_usage"
    )
    store_response("energy", "ac_hours", ac_usage)
    
    heating_usage = st.slider(
        "How many hours did you use heating today?", 
        0, 24, 
        value=int(get_response("energy", "heating_hours", 0)),
        key="heating_usage"
    )
    store_response("energy", "heating_hours", heating_usage)
    
    # Water usage tracking
    st.subheader("Water Usage")
    shower_duration = st.number_input(
        "How many minutes did you shower today?", 
        min_value=0, 
        step=1,
        value=int(get_response("water", "shower_minutes", 0)),
        key="shower_duration"
    )
    store_response("water", "shower_minutes", shower_duration)
    
    # Laundry water and energy usage
    laundry = st.checkbox(
        "Did you do laundry today?",
        value=get_response("water", "did_laundry", False),
        key="did_laundry"
    )
    store_response("water", "did_laundry", laundry)
    
    # Additional laundry questions if applicable
    if laundry:
        laundry_loads = st.number_input(
            "How many loads of laundry?", 
            min_value=1, 
            step=1,
            value=int(get_response("water", "laundry_loads", 1)),
            key="laundry_loads"
        )
        store_response("water", "laundry_loads", laundry_loads)
        
        # Water temperature affects energy usage significantly
        laundry_temp_options = ["Cold", "Warm", "Hot"]
        default_temp = get_response("water", "laundry_temperature", "Cold")
        try:
            default_temp_index = laundry_temp_options.index(default_temp)
        except ValueError:
            default_temp_index = 0
            
        laundry_temp = st.select_slider(
            "At what temperature?", 
            options=laundry_temp_options,
            value=laundry_temp_options[default_temp_index],
            key="laundry_temp"
        )
        store_response("water", "laundry_temperature", laundry_temp)
    
    # Dishwasher usage tracking
    dishwasher = st.checkbox(
        "Did you use a dishwasher today?",
        value=get_response("water", "used_dishwasher", False),
        key="used_dishwasher"
    )
    store_response("water", "used_dishwasher", dishwasher)
    
    # Navigation buttons for this step
    col1, col2, col3 = st.columns([1, 1, 5])
    with col1:
        if st.button("Previous", key="energy_prev", use_container_width=True):
            prev_step()
    with col2:
        if st.button("Next: Consumer Goods", key="energy_next", use_container_width=True):
            next_step()

# Step 4: Consumer Goods - Collects information about purchases and waste management
elif st.session_state.current_step == 4:
    st.header("Consumer Goods")
    
    # Initialize needed categories for consumption and waste
    for category in ["consumption", "waste"]:
        if category not in st.session_state.user_responses:
            st.session_state.user_responses[category] = {}
    
    # Shopping habits and purchases
    purchased_options = ["Clothing", "Electronics", "Furniture", "Books/Media", "Toys", "Household items", "None"]
    default_purchased = get_response("consumption", "purchased_items", [])
    
    purchased_items = st.multiselect(
        "Did you purchase any of the following items today? (Select all that apply)",
        purchased_options,
        default=default_purchased,
        key="purchased_items"
    )
    store_response("consumption", "purchased_items", purchased_items)
    
    # Additional questions if user made purchases
    if "None" not in purchased_items and purchased_items:
        # New vs. second-hand has major impact on footprint
        items_new_options = ["All new", "Mixture of new and second-hand", "All second-hand"]
        default_items_new = get_response("consumption", "items_new_or_used", "All new")
        try:
            default_items_new_index = items_new_options.index(default_items_new)
        except ValueError:
            default_items_new_index = 0
            
        items_new = st.radio(
            "Were these items new or second-hand?", 
            items_new_options,
            index=default_items_new_index,
            key="items_new"
        )
        store_response("consumption", "items_new_or_used", items_new)
        
        # Packaging impacts waste footprint
        packaging_options = ["Minimal/eco-friendly packaging", "Standard packaging", "Excessive packaging"]
        default_packaging = get_response("consumption", "item_packaging", "Standard packaging")
        try:
            default_packaging_index = packaging_options.index(default_packaging)
        except ValueError:
            default_packaging_index = 1
            
        item_packaging = st.radio(
            "How was the packaging of these items?",
            packaging_options,
            index=default_packaging_index,
            key="item_packaging"
        )
        store_response("consumption", "item_packaging", item_packaging)
    
    # Online shopping has transportation impacts
    online_shopping = st.checkbox(
        "Did you order anything online today?",
        value=get_response("consumption", "ordered_online", False),
        key="ordered_online"
    )
    store_response("consumption", "ordered_online", online_shopping)
    
    # Delivery speed affects carbon footprint
    if online_shopping:
        delivery_options = ["Standard", "Express/Next day", "Same day"]
        default_delivery = get_response("consumption", "delivery_option", "Standard")
        try:
            default_delivery_index = delivery_options.index(default_delivery)
        except ValueError:
            default_delivery_index = 0
            
        delivery_option = st.radio(
            "What delivery option did you choose?",
            delivery_options,
            index=default_delivery_index,
            key="delivery_option"
        )
        store_response("consumption", "delivery_option", delivery_option)
    
    # Waste management practices
    st.subheader("Waste & Recycling")
    
    # Recycling percentage significantly impacts waste footprint
    recycling = st.slider(
        "How much of your waste today did you recycle?", 
        0, 100, 
        value=int(get_response("waste", "recycling_percentage", 0)),
        help="Estimated percentage",
        key="recycling"
    )
    store_response("waste", "recycling_percentage", recycling)
    
    # Composting reduces landfill methane emissions
    composting = st.checkbox(
        "Do you compost food waste?",
        value=get_response("waste", "does_compost", False),
        key="does_compost"
    )
    store_response("waste", "does_compost", composting)
    
    # Single-use plastics have high impact relative to their utility
    plastic_usage = st.slider(
        "How many single-use plastic items did you use today?", 
        0, 20, 
        value=int(get_response("waste", "single_use_plastic_count", 0)),
        help="E.g., straws, bags, utensils, packaging",
        key="plastic_usage"
    )
    store_response("waste", "single_use_plastic_count", plastic_usage)
    
    # Navigation buttons for this step
    col1, col2, col3 = st.columns([1, 1, 5])
    with col1:
        if st.button("Previous", key="goods_prev", use_container_width=True):
            prev_step()
    with col2:
        if st.button("Next: Food Invoice", key="goods_next", use_container_width=True):
            next_step()

# Step 5: Food Invoice - Optional step for analyzing food delivery receipts
elif st.session_state.current_step == 5:
    st.header("Food Order Invoice")
    
    # Check if user indicated they have a food delivery invoice
    has_lunch_invoice = get_response("food", "has_lunch_invoice", False)
    has_dinner_invoice = get_response("food", "has_dinner_invoice", False)
    
    # Only show upload if user has an invoice to analyze
    if has_lunch_invoice or has_dinner_invoice:
        st.info("You mentioned you have a food delivery invoice. Please upload it below for analysis.")
        
        # File uploader component with type restrictions
        uploaded_file = st.file_uploader(
            "Upload your food order invoice", 
            type=["jpg", "jpeg", "png", "pdf"],
            key="invoice_upload"
        )
        
        # Process the uploaded file if available
        if uploaded_file is not None:
            # Currently only handling image files, not PDFs
            if uploaded_file.type.startswith('image'):
                # Save the uploaded file locally
                with open("uploaded_invoice.png", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Display the image to the user
                st.image("uploaded_invoice.png", caption="Uploaded Food Invoice", use_container_width=True)
                
                # Provide button to trigger analysis
                analyze_button = st.button("Analyze Invoice", key="analyze_invoice")
                if analyze_button:
                    # Show progress indicator
                    analysis_placeholder = st.empty()
                    analysis_placeholder.warning("Analyzing your food order... This may take up to 60 seconds.")
                    
                    # Encode and send the image for analysis
                    base64_image = encode_image("uploaded_invoice.png")
                    if base64_image:
                        # Use Gemma 3 to analyze the receipt image
                        invoice_analysis = analyze_food_invoice(base64_image)
                        st.session_state.food_order_analysis = invoice_analysis
                        
                        # Update UI with success message
                        analysis_placeholder.success("Analysis complete!")
                        
                        # Display the analysis results
                        st.subheader("Food Order Analysis")
                        st.markdown(invoice_analysis)
                        
                        # Store analysis in user responses
                        store_response("food", "invoice_analysis", invoice_analysis)
                    else:
                        analysis_placeholder.error("Failed to process the image. Please try another image.")
            else:
                st.error("Please upload an image file of your invoice. PDF processing is not currently supported.")
    else:
        st.info("You didn't mention having a food delivery invoice. You can proceed to the next step.")
    
    # Navigation buttons for this step
    col1, col2, col3 = st.columns([1, 1, 5])
    with col1:
        if st.button("Previous", key="invoice_prev", use_container_width=True):
            prev_step()
    with col2:
        if st.button("Next: Calculate Footprint", key="invoice_next", use_container_width=True):
            next_step()

# Step 6: Results - Final calculation and display of carbon footprint
elif st.session_state.current_step == 6:
    st.header("Your Carbon Footprint Results")
    
    # Option to recalculate if needed
    if st.button("Recalculate", key="recalculate"):
        st.session_state.final_result = None
    
    # Display the food invoice analysis if available
    if st.session_state.food_order_analysis:
        st.subheader("Food Order Carbon Footprint")
        st.markdown(st.session_state.food_order_analysis)
        st.markdown("---")
    
    # Calculate overall carbon footprint if not already done
    if st.session_state.final_result is None:
        calculation_placeholder = st.empty()
        calculation_placeholder.info("Preparing to calculate your overall carbon footprint...")
        
        try:
            # Show progress indicator during calculation
            calculation_placeholder.warning("Calculating your carbon footprint... This may take up to 60 seconds.")
            
            # Send all user data to the AI for comprehensive analysis
            result = analyze_with_gemma(st.session_state.user_responses)
            st.session_state.final_result = result
            
            # Update UI with success message
            calculation_placeholder.success("Calculation complete!")
        except Exception as e:
            calculation_placeholder.error(f"Error during calculation: {str(e)}")
            st.session_state.final_result = f"We encountered an error while calculating your carbon footprint: {str(e)}"
    
    # Display the comprehensive analysis results
    st.subheader("Overall Carbon Footprint Analysis")
    st.markdown(st.session_state.final_result)
    
    # Additional recommendations for next steps
    st.subheader("Next Steps")
    st.markdown("""
    - Track your footprint over time to see your progress
    - Set goals to reduce your highest impact areas
    - Share your journey with friends and family to inspire change
    """)
    
    # Navigation and reset buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous", key="results_prev", use_container_width=True):
            prev_step()
    with col2:
        # Start over button clears all session state and reloads the app
        if st.button("Start Over", key="reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
