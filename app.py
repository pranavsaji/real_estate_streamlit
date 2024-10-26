

import os
import pandas as pd
import re
import pandasql as psql
import streamlit as st
import sqlparse
import openai
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
# ============================================
# Helper Functions
# ============================================

import os
from openai import OpenAI


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)
st.session_state.client = client  # Storing in session state if needed

def initialize_openai_client():
    """
    Initializes the OpenAI client using the API key from .env file.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in the .env file.")
        return None
    openai.api_key = api_key
    return "OpenAI client initialized"

def get_openai_completion(
    prompt: str,
    model: str = "gpt-4o",
    max_tokens: int = 1500,
    temperature: float = 0.0,
    stop: Optional[List[str]] = None
) -> str:
    """
    Use OpenAI API to get a completion based on the provided prompt.

    Parameters:
    - prompt (str): The input prompt for the model.
    - model (str): The model to use for completion (default is "gpt-4").
    - max_tokens (int): The maximum number of tokens to generate (default is 1500).
    - temperature (float): Sampling temperature (default is 0.0).
    - stop (Optional[List[str]]): Sequences where the API will stop generating further tokens (default is None).

    Returns:
    - str: The generated completion text.
    """
    # Retrieve the OpenAI client from session state
    client = st.session_state.get('client')

    if client is None:
        st.error("OpenAI client is not initialized. Please ensure that the client is set in session state.")
        return ""

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are ChatGPT, a large language model."},
                {"role": "user", "content": prompt}
            ],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop
        )

        # Validate the response structure
        if (
            response and 
            hasattr(response, 'choices') and 
            len(response.choices) > 0 and 
            hasattr(response.choices[0], 'message') and 
            hasattr(response.choices[0].message, 'content') and
            response.choices[0].message.content
        ):
            return response.choices[0].message.content.strip()
        else:
            st.error("Received an unexpected response structure from OpenAI API.")
            return ""
                
    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")
        return ""


# ============================================
# Caching Mechanism for LLM Responses
# ============================================

def initialize_llm_cache():
    """
    Initializes the LLM cache in session state if it doesn't exist.
    """
    if 'llm_cache' not in st.session_state:
        st.session_state.llm_cache = {}


def get_cached_response(prompt: str, max_tokens: int = 1500, temperature: float = 0.0, stop: Optional[List[str]] = None) -> str:
    """
    Retrieves a cached response for the given prompt if available.
    Otherwise, calls the OpenAI API, caches the response, and returns it.

    Parameters:
    - prompt (str): The input prompt for the model.
    - max_tokens (int): The maximum number of tokens to generate.
    - temperature (float): Sampling temperature.
    - stop (Optional[List[str]]): Stop sequences.

    Returns:
    - str: The generated or cached completion text.
    """
    initialize_llm_cache()
    cache_key = prompt

    if cache_key in st.session_state.llm_cache:
        return st.session_state.llm_cache[cache_key]
    else:
        response = get_openai_completion(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
        st.session_state.llm_cache[cache_key] = response
        return response


# ============================================
# Function to Extract SQL Code from Response
# ============================================
def extract_sql_code(response_text):
    """
    Extract SQL code from the assistant's response.
    Since the response contains only the SQL query, return it directly.
    """
    sql_code = response_text.strip()
    # Optionally, ensure the SQL starts with SELECT
    return sql_code if sql_code.upper().startswith("SELECT") else ""


# ============================================
# Function to Generate SQL Query using OpenAI
# ============================================



def generate_sql_openai(query, user_intent, traits, key_phrases, cities, max_retries=5):
    """
    Generate a SQL query based on the user's natural language query using OpenAI.
    Incorporates user intent, traits, and key phrases into the SQL generation process.
    Attempts up to `max_retries` times if extraction fails.
    
    Parameters:
    - client: Initialized Groq client.
    - query: User's natural language query.
    - user_intent: Extracted user intent.
    - traits: Extracted traits.
    - key_phrases: Extracted key phrases.
    - cities: List of unique cities from the CSV.
    - max_retries: Number of retry attempts for generating SQL.
    
    Returns:
    - SQL query string or empty string if failed.
    """
    # Convert cities list to a readable string
    cities_str = ", ".join([f"'{city}'" for city in cities])
    print("unique cities are", cities_str)
    prompt = (
        "You are a SQL assistant specialized in real estate data. "
        "Based on the user's natural language query, user intent, traits, and key phrases, generate an accurate SQL query to search the properties. "
        "City and other dtring columns should be matched using LIKE instead of IN"
        "Map any broad location terms in the user's query to matching cities from the Available cities below . "
        "Ensure that all property features and filters mentioned in the query are accurately represented in the SQL. "
        "Use the following CSV columns for the 'zillow_data' table: 'price', 'beds', 'baths', 'area', 'listing_agent', 'year_built', "
        "'property_tax', 'school_ratings', 'neighborhood_desc', 'broker', 'city', 'state', 'zip_code',hoa_fees "
        "Use LIKE operators for string matching and ensure numeric comparisons are correctly handled. "
        "If the user specifies a broad location like 'Bay Area,' map it to matching cities present in Available cities below "
        "If some location or keyword is incomplete, fill them with the most appropriate value from the data, e.g., replace 'redwood' with 'Redwood City'. "
        "If no location is provided, don't add it as a filtering criterion. "
        "If certain values are not provided in the query, don't add them in the SQL query. "
        "If school rating is given as good in the query, use SQL condition as school_ratings >= 7. "
        "If school rating is given as excellent in the query, use SQL condition as school_ratings >= 8. "
        "When multiple conditions exist within the same column, combine them using OR and enclose them in parentheses. "
        "Combine conditions across different columns using AND. "
        "IMPORTANT: For all property features (e.g., 'pool', 'sea_view', 'hoa_fees', 'gym', 'rooftop_access', 'home_theater', 'wine_cellar', 'large_garden', 'high_ceiling', 'hardwood_floors', 'finished_basement', 'garage', 'exposed_brick_walls', 'spacious_backyard', 'solar_panels', 'panoramic_city_views', 'private_elevator', 'fireplace', 'swimming_pool'.etc), search within the 'neighborhood_desc' column using LIKE operators instead of using dedicated feature columns. "
        "Do NOT use separate columns like 'pool', 'gym', etc., for filtering features. Instead, encapsulate all feature-related filters within 'neighborhood_desc'. "
        "Return only the SQL query as plain text without any explanations, code fences, backticks, markdown formatting, or additional text.\n\n"

        "### Available Cities:\n"
        f"{cities_str}\n\n"

        "### Example 1:\n"
        "**User Intent:** The user is searching for a spacious three-bedroom house in San Francisco, prioritizing properties with large backyards.\n"
        "**Traits:**\n"
        "- is a house\n"
        "  has 3 bed, 2 bath\n"
        "  is in San Francisco\n"
        "- has a big backyard\n"
        "**Key Phrases:**\n"
        "3 bedroom house\n"
        "big backyard\n"
        "San Francisco real estate\n"
        "spacious home\n"
        "outdoor space\n"
        "family-friendly neighborhood\n"
        "gardening space\n"
        "entertainment area\n"
        "pet-friendly home\n"
        "modern amenities\n\n"
        "**User Query:** \"Looking for a 3 bedroom, 2 bathroom house with a big backyard in San Francisco.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 3 AND baths = 2 AND city LIKE '%San Francisco%' AND (neighborhood_desc LIKE '%big backyard%');\n\n"

        
    
        "### Example 3:\n"
        "**User Intent:** The user wants a four-bedroom house in Miami with a pool and sea view, under a budget of $2 million.\n"
        "**Traits:**\n"
        "- is a house\n"
        "  has 4 bed, 3 bath\n"
        "  is in Miami\n"
        "- has a pool\n"
        "- has a sea view\n"
        "- is priced under $2,000,000\n"
        "**Key Phrases:**\n"
        "4 bedroom house\n"
        "pool\n"
        "sea view\n"
        "Miami real estate\n"
        "luxury home\n"
        "waterfront property\n"
        "family-friendly neighborhood\n"
        "modern design\n"
        "spacious backyard\n"
        "gated community\n\n"
        "**User Query:** \"Looking for a 4 bed 3 bath house in Miami with a pool and sea view, priced under 2 million.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 4 AND baths = 3 AND price <= 2000000 AND city LIKE '%Miami%' AND (neighborhood_desc LIKE '%pool%' OR neighborhood_desc LIKE '%sea view%');\n\n"

        "### Example 4:\n"
        "**User Intent:** The user is searching for a three-bedroom townhouse in Denver with low HOA fees and property taxes, priced below $700,000.\n"
        "**Traits:**\n"
        "- is a townhouse\n"
        "  has 3 bed, 2 bath\n"
        "  is in Denver\n"
        "- has low HOA fees\n"
        "- has low property taxes\n"
        "- is priced under $700,000\n"
        "**Key Phrases:**\n"
        "3 bedroom townhouse\n"
        "low HOA fees\n"
        "low property taxes\n"
        "Denver real estate\n"
        "budget-friendly\n"
        "family-friendly\n"
        "spacious living\n"
        "modern amenities\n"
        "central location\n"
        "pet-friendly\n\n"
        "**User Query:** \"Looking for a 3 bed 2 bath townhouse in Denver with low HOA fees and property taxes, priced under 700,000.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 3 AND baths = 2 AND price <= 700000 AND city LIKE '%Denver%' AND (neighborhood_desc LIKE '%low HOA fees%' OR neighborhood_desc LIKE '%low property taxes%');\n\n"

        "### Example 5:\n"
        "**User Intent:** The user wants a two-bedroom condo in San Francisco with a gym and rooftop access, priced below $1.2 million.\n"
        "**Traits:**\n"
        "- is a condo\n"
        "  has 2 bed, 2 bath\n"
        "  is in San Francisco\n"
        "- has a gym\n"
        "- has rooftop access\n"
        "- is priced under $1,200,000\n"
        "**Key Phrases:**\n"
        "2 bedroom condo\n"
        "gym\n"
        "rooftop access\n"
        "San Francisco real estate\n"
        "modern amenities\n"
        "urban living\n"
        "secure building\n"
        "pet-friendly\n"
        "spacious interiors\n"
        "high-rise building\n\n"
        "**User Query:** \"Looking for a 2 bed 2 bath condo in San Francisco with a gym and rooftop access, priced under 1.2 million.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 2 AND baths = 2 AND price <= 1200000 AND city LIKE '%San Francisco%' AND (neighborhood_desc LIKE '%gym%' OR neighborhood_desc LIKE '%rooftop access%');\n\n"

        "### Example 6:\n"
        "**User Intent:** The user is searching for a five-bedroom villa in Los Angeles with a home theater, wine cellar, and large garden, priced below $3 million.\n"
        "**Traits:**\n"
        "- is a villa\n"
        "  has 5 bed, 4 bath\n"
        "  is in Los Angeles\n"
        "- has a home theater\n"
        "- has a wine cellar\n"
        "- has a large garden\n"
        "- is priced under $3,000,000\n"
        "**Key Phrases:**\n"
        "5 bedroom villa\n"
        "home theater\n"
        "wine cellar\n"
        "large garden\n"
        "Los Angeles real estate\n"
        "luxury amenities\n"
        "spacious property\n"
        "gated community\n"
        "waterfront property\n"
        "exclusive neighborhood\n\n"
        "**User Query:** \"Looking for a 5 bed 4 bath villa in Los Angeles with a home theater, wine cellar, and large garden, priced under 3 million.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 5 AND baths = 4 AND price <= 3000000 AND city LIKE '%Los Angeles%' AND (neighborhood_desc LIKE '%home theater%' OR neighborhood_desc LIKE '%wine cellar%' OR neighborhood_desc LIKE '%large garden%');\n\n"

        "### Example 7:\n"
        "**User Intent:** The user wants a studio apartment in Redwood City with high ceilings and hardwood floors, priced below $900,000.\n"
        "**Traits:**\n"
        "- is a studio apartment\n"
        "  has 1 bed, 1 bath\n"
        "  is in Redwood City\n"
        "- has high ceilings\n"
        "- has hardwood floors\n"
        "- is priced under $900,000\n"
        "**Key Phrases:**\n"
        "studio apartment\n"
        "high ceilings\n"
        "hardwood floors\n"
        "Redwood City real estate\n"
        "modern design\n"
        "urban living\n"
        "pet-friendly\n"
        "open floor plan\n"
        "luxury finishes\n"
        "secure building\n\n"
        "**User Query:** \"Looking for a studio apartment in Redwood with high ceilings and hardwood floors, priced under 900,000.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 1 AND baths = 1 AND price <= 900000 AND city LIKE '%Redwood City%' AND (neighborhood_desc LIKE '%high ceilings%' OR neighborhood_desc LIKE '%hardwood floors%');\n\n"

        "### Example 8:\n"
        "**User Intent:** The user is searching for a three-bedroom duplex in Chicago with a finished basement and garage, priced below $1.5 million.\n"
        "**Traits:**\n"
        "- is a duplex\n"
        "  has 3 bed, 2 bath\n"
        "  is in Chicago\n"
        "- has a finished basement\n"
        "- has a garage\n"
        "- is priced under $1,500,000\n"
        "**Key Phrases:**\n"
        "3 bedroom duplex\n"
        "finished basement\n"
        "garage\n"
        "Chicago real estate\n"
        "spacious living\n"
        "family-friendly neighborhood\n"
        "modern amenities\n"
        "secure property\n"
        "close to schools\n"
        "well-maintained\n\n"
        "**User Query:** \"Looking for a 3 bed 2 bath duplex in Chicago with a finished basement and garage, priced under 1.5 million.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 3 AND baths = 2 AND price <= 1500000 AND city LIKE '%Chicago%' AND (neighborhood_desc LIKE '%finished basement%' OR neighborhood_desc LIKE '%garage%');\n\n"

        "### Example 9:\n"
        "**User Intent:** The user wants a two-bedroom loft in Boston with exposed brick walls and large windows, priced below $1.1 million.\n"
        "**Traits:**\n"
        "- is a loft\n"
        "  has 2 bed, 1 bath\n"
        "  is in Boston\n"
        "- has exposed brick walls\n"
        "- has large windows\n"
        "- is priced under $1,100,000\n"
        "**Key Phrases:**\n"
        "2 bedroom loft\n"
        "exposed brick walls\n"
        "large windows\n"
        "Boston real estate\n"
        "industrial design\n"
        "open floor plan\n"
        "modern amenities\n"
        "urban living\n"
        "pet-friendly\n"
        "high ceilings\n\n"
        "**User Query:** \"Looking for a 2 bed 1 bath loft in Boston with exposed brick walls and large windows, priced under 1.1 million.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 2 AND baths = 1 AND price <= 1100000 AND city LIKE '%Boston%' AND (neighborhood_desc LIKE '%exposed brick walls%' OR neighborhood_desc LIKE '%large windows%');\n\n"

        "### Example 10:\n"
        "**User Intent:** The user is searching for a four-bedroom ranch-style house in Phoenix with a spacious backyard and solar panels, priced below $850,000.\n"
        "**Traits:**\n"
        "- is a ranch-style house\n"
        "  has 4 bed, 3 bath\n"
        "  is in Phoenix\n"
        "- has a spacious backyard\n"
        "- has solar panels\n"
        "- is priced under $850,000\n"
        "**Key Phrases:**\n"
        "4 bedroom ranch-style house\n"
        "spacious backyard\n"
        "solar panels\n"
        "Phoenix real estate\n"
        "energy-efficient\n"
        "family-friendly neighborhood\n"
        "modern amenities\n"
        "secure property\n"
        "low maintenance\n"
        "pet-friendly\n\n"
        "**User Query:** \"Looking for a 4 bed 3 bath ranch-style house in Phoenix with a spacious backyard and solar panels, priced under 850,000.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 4 AND baths = 3 AND price <= 850000 AND city LIKE '%Phoenix%' AND (neighborhood_desc LIKE '%spacious backyard%' OR neighborhood_desc LIKE '%solar panels%');\n\n"

        "### Example 11:\n"
        "**User Intent:** The user wants a three-bedroom bungalow in Portland with a gourmet kitchen and hardwood floors, priced below $950,000.\n"
        "**Traits:**\n"
        "- is a bungalow\n"
        "  has 3 bed, 2 bath\n"
        "  is in Portland\n"
        "- has a gourmet kitchen\n"
        "- has hardwood floors\n"
        "- is priced under $950,000\n"
        "**Key Phrases:**\n"
        "3 bedroom bungalow\n"
        "gourmet kitchen\n"
        "hardwood floors\n"
        "Portland real estate\n"
        "modern amenities\n"
        "family-friendly neighborhood\n"
        "spacious living areas\n"
        "secure property\n"
        "pet-friendly\n"
        "well-maintained\n\n"
        "**User Query:** \"Looking for a 3 bed 2 bath bungalow in Portland with a gourmet kitchen and hardwood floors, priced under 950,000.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 3 AND baths = 2 AND price <= 950000 AND city LIKE '%Portland%' AND (neighborhood_desc LIKE '%gourmet kitchen%' OR neighborhood_desc LIKE '%hardwood floors%');\n\n"

        "### Example 12:\n"
        "**User Intent:** The user is searching for a duplex in Houston with energy-efficient appliances and a home office, priced below $1.3 million.\n"
        "**Traits:**\n"
        "- is a duplex\n"
        "  has 2 bed, 2 bath per unit\n"
        "  is in Houston\n"
        "- has energy-efficient appliances\n"
        "- has a home office\n"
        "- is priced under $1,300,000\n"
        "**Key Phrases:**\n"
        "duplex\n"
        "energy-efficient appliances\n"
        "home office\n"
        "Houston real estate\n"
        "modern amenities\n"
        "spacious interiors\n"
        "family-friendly neighborhood\n"
        "secure property\n"
        "pet-friendly\n"
        "well-maintained\n\n"
        "**User Query:** \"Looking for a duplex in Houston with energy-efficient appliances and a home office, priced under 1.3 million.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE type = 'duplex' AND price <= 1300000 AND city LIKE '%Houston%' AND (neighborhood_desc LIKE '%energy-efficient appliances%' OR neighborhood_desc LIKE '%home office%');\n\n"

        "### Example 13:\n"
        "**User Intent:** The user wants a five-bedroom farmhouse in Nashville with a barn, landscaped garden, and solar energy system, priced below $2.2 million.\n"
        "**Traits:**\n"
        "- is a farmhouse\n"
        "  has 5 bed, 4 bath\n"
        "  is in Nashville\n"
        "- has a barn\n"
        "- has a landscaped garden\n"
        "- has a solar energy system\n"
        "- is priced under $2,200,000\n"
        "**Key Phrases:**\n"
        "5 bedroom farmhouse\n"
        "barn\n"
        "landscaped garden\n"
        "solar energy system\n"
        "Nashville real estate\n"
        "luxury amenities\n"
        "spacious property\n"
        "family-friendly neighborhood\n"
        "modern amenities\n"
        "energy-efficient\n\n"
        "**User Query:** \"Looking for a 5 bed 4 bath farmhouse in Nashville with a barn, landscaped garden, and solar energy system, priced under 2.2 million.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 5 AND baths = 4 AND price <= 2200000 AND city LIKE '%Nashville%' AND (neighborhood_desc LIKE '%barn%' OR neighborhood_desc LIKE '%landscaped garden%' OR neighborhood_desc LIKE '%solar energy system%');\n\n"

        "### Example 14:\n"
        "**User Intent:** The user is searching for a two-bedroom penthouse in Dallas with panoramic city views and a private elevator, priced below $1.8 million.\n"
        "**Traits:**\n"
        "- is a penthouse\n"
        "  has 2 bed, 2 bath\n"
        "  is in Dallas\n"
        "- has panoramic city views\n"
        "- has a private elevator\n"
        "- is priced under $1,800,000\n"
        "**Key Phrases:**\n"
        "2 bedroom penthouse\n"
        "panoramic city views\n"
        "private elevator\n"
        "Dallas real estate\n"
        "luxury amenities\n"
        "high-rise living\n"
        "secure building\n"
        "modern design\n"
        "spacious interiors\n"
        "pet-friendly\n\n"
        "**User Query:** \"Looking for a 2 bed 2 bath penthouse in Dallas with panoramic city views and a private elevator, priced under 1.8 million.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 2 AND baths = 2 AND price <= 1800000 AND city LIKE '%Dallas%' AND (neighborhood_desc LIKE '%panoramic city views%' OR neighborhood_desc LIKE '%private elevator%');\n\n"

        "### Example 15:\n"
        "**User Intent:** The user wants a three-bedroom colonial-style house in Philadelphia with a fireplace, home theater, and swimming pool, priced below $1.4 million.\n"
        "**Traits:**\n"
        "- is a colonial-style house\n"
        "  has 3 bed, 2 bath\n"
        "  is in Philadelphia\n"
        "- has a fireplace\n"
        "- has a home theater\n"
        "- has a swimming pool\n"
        "- is priced under $1,400,000\n"
        "**Key Phrases:**\n"
        "3 bedroom colonial-style house\n"
        "fireplace\n"
        "home theater\n"
        "swimming pool\n"
        "Philadelphia real estate\n"
        "modern amenities\n"
        "spacious backyard\n"
        "family-friendly neighborhood\n"
        "secure property\n"
        "pet-friendly\n\n"
        "**User Query:** \"Looking for a 3 bed 2 bath colonial-style house in Philadelphia with a fireplace, home theater, and swimming pool, priced under 1.4 million.\"\n\n"
        "**SQL Query:**\n"
        "SELECT * FROM zillow_data WHERE beds = 3 AND baths = 2 AND price <= 1400000 AND city LIKE '%Philadelphia%' AND (neighborhood_desc LIKE '%fireplace%' OR neighborhood_desc LIKE '%home theater%' OR neighborhood_desc LIKE '%swimming pool%');\n\n"

        "---\n\n"
        "**User Intent:**\n"
        f"{user_intent}\n\n"
        "**Traits:**\n"
        f"{traits}\n\n"
        "**Key Phrases:**\n"
        f"{key_phrases}\n\n"
        "**User Query:**\n"
        f"\"{query}\"\n\n"
        "**SQL Query:**"
    )

    for attempt in range(1, max_retries + 1):
        response = get_cached_response(prompt, max_tokens=1500, temperature=0)
        sql_query = extract_sql_code(response)
        if sql_query:
            return sql_query.strip()
        else:
            st.warning(f"Attempt {attempt} to extract SQL code failed.")

    # After all retries failed
    st.error("Could not generate a valid SQL query after multiple attempts.")
    return ""


# ============================================
# Function to Generate Property Keywords using OpenAI
# ============================================
def get_property_keywords_openai(query, user_intent, traits, key_phrases, sql_query):
    """
    Generate PropertyKeywords based on the query, user intent, traits, key phrases, and SQL query using OpenAI.
    
    Parameters:
    - query: User's natural language query.
    - user_intent: Extracted user intent.
    - traits: Extracted traits.
    - key_phrases: Extracted key phrases.
    - sql_query: Generated SQL query.
    
    Returns:
    - PropertyKeywords string or empty string if extraction fails.
    """
    prompt = (
        "Analyze the following real estate query, user intent, traits, key phrases, and SQL query to extract the values used for each column in the SQL statement."
        " Do not miss any content or value from SQL."
        " Format the output as a comma-separated list in the format 'Column: Value'. But don't add any extra piece of text."
        " Ensure that each value corresponds accurately to the SQL query."
        " IMPORTANT: Only give precise output in the format given without any additional text or tokens."
        " If some location or keyword is incomplete, fill them with the most appropriate value from the data, e.g., replace 'redwood' with 'Redwood City'."
        " Do not include any explanations or additional text.\n\n"

        "### Query:\n"
        f"\"{query}\"\n\n"

        "### User Intent:\n"
        f"{user_intent}\n\n"

        "### Traits:\n"
        f"{', '.join(traits)}\n\n"

        "### Key Phrases:\n"
        f"{', '.join(key_phrases)}\n\n"

        "### SQL Query:\n"
        f"{sql_query}\n\n"

        "### PropertyKeywords:"
    )

    response = get_cached_response(prompt, temperature=0, max_tokens=700)
    return response.strip()


# ============================================
# Function to Extract User Intent using OpenAI
# ============================================
def get_user_intent_openai(query):
    """
    Extract the user intent from the input query using OpenAI with K-shot prompting.
    """
    prompt = (
        "Analyze the following real estate query and extract the user intent. "
        "Provide the intent as a concise paragraph without any additional text or explanations."
        "If some location or keyword is incomplete, fill them with the most appropriate value from the data, e.g., replace 'redwood' with 'Redwood City'.\n\n"
        "### Example 1:\n"
        "**Query:** \"Looking for a 3 bedroom house with a big backyard in San Francisco.\"\n"
        "**User Intent:** The user is searching for a spacious three-bedroom house in San Francisco, prioritizing properties with large backyards. They likely value outdoor space for activities such as gardening or entertaining.\n\n"
        "### Example 2:\n"
        "**Query:** \"Seeking a 2 bedroom apartment near downtown Seattle with modern amenities.\"\n"
        "**User Intent:** The user is interested in a two-bedroom apartment near downtown Seattle, emphasizing modern amenities. They likely prioritize convenience and contemporary living spaces.\n\n"
        "### Example 3:\n"
        "**Query:** \"2 bed 2 bath in Irvine and 3 bed 2 bath in Redwood under 1600000.\"\n"
        "**User Intent:** The user is looking for both a 2-bedroom, 2-bathroom house in Irvine and a 3-bedroom, 2-bathroom property in Redwood City, with a combined budget under 1,600,000. They seek multiple options within a specific price range.\n\n"
        "### Example 4:\n"
        "**Query:** \"Looking for a 4-bedroom villa in Redwood with a pool and sea view, priced below 2 million.\"\n"
        "**User Intent:** The user desires a luxurious four-bedroom villa in Redwood City that includes a pool and offers a sea view, with a budget below 2 million. They prioritize luxury and scenic views.\n\n"
        "### Example 5:\n"
        "**Query:** \"Searching for 1 bed 1 bath condo in Redwood and 2 bed 2 bath townhouse in Boston under 750000.\"\n"
        "**User Intent:** The user is seeking both a 1-bedroom, 1-bathroom condo in Redwood City and a 2-bedroom, 2-bathroom townhouse in Boston, with a maximum budget of 750,000. They are interested in multiple property types across different cities within a specified price range.\n\n"
        "---\n\n"
        "**User Intent:**\n"
        f"{query}\n"
        "**User Intent:**"
    )
    response = get_cached_response(prompt)
    return response.strip()


# ============================================
# Function to Extract Traits using OpenAI
# ============================================
def get_traits_openai(query, user_intent):
    """
    Extract key traits from the input query using OpenAI with K-shot prompting.
    Each trait includes a verb phrase and follows a nested bullet structure.
    """
    prompt = (
        "From the following real estate query and user intent, extract the key traits."
        " Provide each trait starting with a verb phrase without any explanations or additional text."
        " Ensure that each trait is concise and relevant to the user's request."
        " Do not split numerical values like prices across multiple lines."
        " If multiple properties are mentioned, list each property separately and combine the budget where applicable."
        " Do not include any preamble, emojis, or phrases like 'Here are the extracted traits:'."
        " If it's not explicitly mentioned as a house or property, don't add that as a trait."
        " If some location or keyword is incomplete, fill them with the most appropriate value from the data, e.g., replace 'redwood' with 'Redwood City'."
        " Your response should only include the traits, exactly as in the examples, and nothing else.\n\n"
        "---\n\n"
        "**Example 1:**\n\n"
        "**User Intent:** The user is searching for a spacious three-bedroom house in San Francisco, prioritizing properties with large backyards. They likely value outdoor space for activities such as gardening or entertaining.\n\n"
        "**Query:** \"Looking for a 3 bedroom, 2 bathroom house with a big backyard in San Francisco.\"\n\n"
        "**Traits:**\n"
        "    is a house\n"
        "    has 3 bed, 2 bath\n"
        "    is in San Francisco\n"
        "    has a big backyard\n\n"
        "**Example 2:**\n\n"
        "**User Intent:** The user is interested in a two-bedroom apartment near downtown Seattle, emphasizing modern amenities. They likely prioritize convenience and contemporary living spaces.\n\n"
        "**Query:** \"Seeking a 2 bed 1 bath condo in downtown Seattle with modern amenities.\"\n\n"
        "**Traits:**\n"
        "    is a condo\n"
        "    has 2 bed, 1 bath\n"
        "    is in Seattle\n"
        "    has modern amenities\n\n"
        "**Example 3:**\n\n"
        "**User Intent:** The user is interested in finding a 2-bedroom, 2-bathroom property in Irvine and a 3-bedroom, 3-bathroom property in San Francisco, with a combined budget of $1,595,000.\n\n"
        "**Query:** \"2 bed 2 bath in Irvine and 3 bed 3 bath in San Francisco both under 1,595,000.\"\n\n"
        "**Traits:**\n"
        "    has 2 bed, 2 bath\n"
        "    is in Irvine\n"
        "    has 3 bed, 3 bath\n"
        "    is in San Francisco\n"
        "    is under $1,595,000.\n\n"
        "---\n\n"
        "**User Intent:**\n"
        f"{user_intent}\n\n"
        "**Query:**\n"
        f"\"{query}\"\n\n"
        "**Traits:**"
    )

    response = get_cached_response(prompt, temperature=0, max_tokens=1500, stop=["\n\n"])
    if response:
        traits_list = [line.strip() for line in response.splitlines() if line.strip()]
        return traits_list
    return []


# ============================================
# Function to Extract Key Phrases using OpenAI
# ============================================
def get_key_phrases_openai(query, user_intent, traits):
    """
    Extract top key phrases from the input query, user intent, and traits using OpenAI with K-shot prompting.
    """
    prompt = (
        "From the following real estate query, user intent, and traits, extract the top most relevant key phrases that can be used for search optimization or listing purposes."
        " Provide each key phrase on a separate line without any explanations or additional text."
        " Do not include any preamble, emojis, or phrases like 'Here are the top most relevant key phrases for search optimization or listing purposes:'."
        " If some location or keyword is incomplete, fill them with the most appropriate value from the data, e.g., replace 'redwood' with 'Redwood City'."
        " Your response should only include the key phrases, and structure it exactly as in the examples, and add no extra tokens.\n\n"
        "---\n\n"
        "**Example 1:**\n\n"
        "**User Intent:** The user is searching for a spacious three-bedroom house in San Francisco, prioritizing properties with large backyards. They likely value outdoor space for activities such as gardening or entertaining.\n\n"
        "**Traits:**\n"
        "    is a house\n"
        "    has 3 bed, 2 bath\n"
        "    is in San Francisco\n"
        "    has a big backyard\n\n"
        "**Query:** \"Looking for a 3 bedroom, 2 bathroom house with a big backyard in San Francisco.\"\n\n"
        "**Key Phrases:**\n"
        "3 bedroom house\n"
        "big backyard\n"
        "San Francisco real estate\n"
        "spacious home\n"
        "outdoor space\n"
        "family-friendly neighborhood\n"
        "gardening space\n"
        "entertainment area\n"
        "pet-friendly home\n"
        "modern amenities\n\n"
        "**Example 2:**\n\n"
        "**User Intent:** The user is interested in a two-bedroom apartment near downtown Seattle, emphasizing modern amenities. They likely prioritize convenience and contemporary living spaces.\n\n"
        "**Traits:**\n"
        "    is a condo\n"
        "    has 2 bed, 1 bath\n"
        "    is in Seattle\n"
        "    has modern amenities\n\n"
        "**Query:** \"Seeking a 2 bed 1 bath condo in downtown Seattle with modern amenities.\"\n\n"
        "**Key Phrases:**\n"
        "2 bedroom apartment\n"
        "downtown Seattle\n"
        "modern amenities\n"
        "urban living\n"
        "convenient location\n"
        "contemporary design\n"
        "city views\n"
        "public transportation access\n"
        "stylish interiors\n"
        "efficient layout\n\n"
        "---\n\n"
        "**User Intent:**\n"
        f"{user_intent}\n\n"
        "**Traits:**\n"
        f"{', '.join(traits)}\n\n"
        "**Query:**\n"
        f"\"{query}\"\n\n"
        "**Key Phrases:**"
    )

    response = get_cached_response(prompt, temperature=0, max_tokens=1000, stop=["\n\n"])
    if response:
        key_phrases_list = [line.strip() for line in response.splitlines() if line.strip()]
        return key_phrases_list[:5]
    return []


# ============================================
# Function to Extract User Intent, Traits, and Key Phrases using OpenAI
# ============================================
def extract_information(query):
    """
    Extract user intent, traits, and key phrases from the input query using OpenAI.
    """
    user_intent = get_user_intent_openai(query)
    traits = get_traits_openai(query, user_intent)
    key_phrases = get_key_phrases_openai(query, user_intent, traits)
    return user_intent, traits, key_phrases


# ============================================
# Function to Execute SQL Query on Pandas DataFrame
# ============================================
def execute_sql_query(sql_query, df):
    """
    Execute the generated SQL query on the Pandas DataFrame using pandasql.
    """
    try:
        # Define 'zillow_data' in the local scope for pandasql
        zillow_data = df

        # Execute the SQL query on the DataFrame
        result = psql.sqldf(sql_query, locals())

        return result
    except Exception as e:
        st.error(f"Error executing SQL query: {e}")
        return None


# ============================================
# Function to Save All Outputs to a Text File
# ============================================
def save_to_txt(file_name, query, user_intent, traits, key_phrases, property_keywords, sql_query, final_output):
    """
    Save the input query, user intent, traits, key phrases, PropertyKeywords, SQL query, and final output to a text file.
    """
    with open(file_name, 'w') as file:
        file.write(f"Input Query: {query}\n\n")
        file.write(f"User Intent: {user_intent}\n\n")
        file.write(f"Traits: {', '.join(traits)}\n\n")
        file.write(f"Key Phrases: {', '.join(key_phrases)}\n\n")
        file.write(f"PropertyKeywords: {property_keywords}\n\n")
        file.write(f"Generated SQL Query: {sql_query}\n\n")
        if final_output is not None and not final_output.empty:
            file.write(f"Final Output from CSV:\n{final_output.to_string(index=False)}\n\n")
        else:
            file.write("Final Output from CSV: No results found.\n\n")


def truncate_text(text, max_length=30):
    """
    Truncate the text to a maximum length and append '...' if truncated.
    
    Parameters:
    - text (str): The original text to truncate.
    - max_length (int): The maximum allowed length of the text.
    
    Returns:
    - str: The truncated text with '...' appended if it was longer than max_length.
    """
    return text if len(text) <= max_length else text[:max_length].rstrip() + '...'


# ============================================
# Initialize Session State Variables
# ============================================
def initialize_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 'page1'
    if 'query' not in st.session_state:
        st.session_state.query = ''
    if 'user_intent' not in st.session_state:
        st.session_state.user_intent = ''
    if 'traits' not in st.session_state:
        st.session_state.traits = []
    if 'key_phrases' not in st.session_state:
        st.session_state.key_phrases = []
    if 'property_keywords' not in st.session_state:
        st.session_state.property_keywords = ''
    if 'sql_query' not in st.session_state:
        st.session_state.sql_query = ''
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'client' not in st.session_state:
        st.session_state.client = initialize_openai_client()
    if 'df' not in st.session_state:
        st.session_state.df = load_data()
    if 'broker_df' not in st.session_state:
        st.session_state.broker_df = load_broker_data()
    if 'selected_broker' not in st.session_state:
        st.session_state.selected_broker = None
    if 'unique_cities' not in st.session_state:
        st.session_state.unique_cities = get_unique_cities(st.session_state.df) if st.session_state.df is not None else []
    
    # Initialize previous searches
    if 'previous_searches' not in st.session_state:
        st.session_state.previous_searches = []
    
    # Initialize selected_zip_code
    if 'selected_zip_code' not in st.session_state:
        st.session_state.selected_zip_code = None


# ============================================
# Navigation Functions with Callbacks
# ============================================
def go_to_page2():
    st.session_state.page = 'page2'
    st.rerun()


def go_to_page3():
    st.session_state.page = 'page3'
    st.rerun()


def go_to_page1():
    st.session_state.page = 'page1'
    st.rerun()


def go_to_page4():
    st.session_state.page = 'page4'
    st.rerun()



def get_column_name_from_trait(trait):
    """
    Generate a consistent, concise, and human-readable column name from a given trait using OpenAI.
    
    Parameters:
    - trait (str): The trait description.
    
    Returns:
    - str: A concise and suitable column name with natural language formatting, ensuring consistency.
    """
    # print("Trait is", trait)
    prompt = (
        "Convert the following real estate trait into a concise, human-readable, and consistent column name "
        "suitable for a real estate data table. Ensure that traits involving numbers (e.g., '2 bed 2 bath' and "
        "'3 bed, 2 bath') follow the same format, and price-related traits are formatted consistently. "
        " Make sure to not miss any trait."
        "Capitalize proper nouns, and keep the output clean without unnecessary punctuation or extra words. "
        "Respond with only the column name without any additional text or explanations.\n\n"
        
        "Examples:\n"
        "Trait: \"is in Bay Area\"\n"
        "Column Name: Bay Area\n\n"
        
        "Trait: \"has a pool\"\n"
        "Column Name: Pool\n\n"
        
        "Trait: \"is near a restaurant\"\n"
        "Column Name: Restaurant nearby\n\n"
        
        "Trait: \"is near a well-rated school\"\n"
        "Column Name: Good school nearby\n\n"
        
        "Trait: \"Price under 1600000\"\n"
        "Column Name: Price under $1,600,000\n\n"
        
        "Trait: \"2 bed 2 bath\"\n"
        "Column Name: 2 bed, 2 bath\n\n"
        
        "Trait: \"3 bed 2 bath\"\n"
        "Column Name: 3 bed, 2 bath\n\n"
        
        f"Trait: \"{trait}\"\n"
        "Column Name:"
    )
    
    column_name = get_cached_response(prompt, max_tokens=15, temperature=0.0)
    print("Column name is",column_name )

    # Post-processing for additional consistency
    if column_name:
        column_name = column_name.strip()
        
        # Ensure consistency in bed/bath formatting (always include a comma)
        column_name = re.sub(r'(\d+ bed)\s+(\d+ bath)', r'\1, \2', column_name)

        # Ensure proper capitalization and trimming of extra spaces
        column_name = column_name.strip().capitalize()

        return column_name
    else:
        # Fallback: Return the trait as a capitalized, more readable version
        return trait.capitalize()




def is_trait_matched(property_row, trait):
    """
    Determine if a property matches a given trait using OpenAI.

    Parameters:
    - property_row (pd.Series): A row from the DataFrame representing a property.
    - trait (str): The trait to check against the property.

    Returns:
    - str: 'yes', 'no', or 'unsure' based on the property trait match.
    """
    # Extract all relevant columns
    relevant_columns = [
        'price', 'beds', 'baths', 'area', 'listing_agent', 'year_built',
        'property_tax', 'school_ratings', 'neighborhood_desc',
        'broker', 'city', 'state', 'zip_code', 'hoa_fees'
    ]

    # Construct a detailed description of the property
    property_details = "\n".join([
        f"{col.replace('_', ' ').title()}: {property_row.get(col, 'N/A')}"
        for col in relevant_columns
    ])

    # Define the list of allowed cities
    allowed_cities = [
        'Atherton', 'Bronx', 'Brooklyn', 'Far Rockaway', 'Flushing',
        'Irvine', 'Jamaica', 'Maspeth', 'Menlo Park', 'Middle Village',
        'Ozone Park', 'Queens Village', 'Redwood City', 'Richmond Hill North',
        'Rosedale', 'San Carlos', 'San Francisco', 'Staten Island',
        'Tustin', 'Woodhaven'
    ]

    # Prepare the prompt with clear instructions and examples
    prompt = (
        "You are an intelligent assistant specialized in real estate analysis. "
        "Given the detailed information about a property and a specific trait, determine whether the property satisfies the trait. "
        "If a property value is closely related to a trait even though it's not an exact match (e.g., 'redwood' instead of 'Redwood City'), map it accordingly. "
        "Respond with 'yes' if the trait fully matches, 'no' if it does not match, and 'unsure' if the information is incomplete or only partially matches.\n\n"
        "### Property Details:\n"
        f"{property_details}\n\n"
        "### Trait to Evaluate:\n"
        f"{trait}\n\n"
        "### Guidelines:\n"
        "- Respond with only one of the following options: 'yes', 'no', or 'unsure'. Do not include any additional text.\n"
        "- If the trait is clearly satisfied by the property details, respond with 'yes'.\n"
        "- If the trait is clearly not satisfied by the property details, respond with 'no'.\n"
        "- If the property details lack sufficient information to determine the trait, or if the match is partial, respond with 'unsure'.\n"
        "- For city-related traits, map any partial or misspelled city names to the correct full name from the allowed cities list.\n\n"
        "### Allowed Cities:\n"
        f"{', '.join(allowed_cities)}\n\n"
        "### Examples:\n\n"
        "#### Example 1:\n"
        "**Property Details:**\n"
        "Price: 850000\n"
        "Beds: 3\n"
        "Baths: 2\n"
        "Area: 2000 sqft\n"
        "Listing Agent: Jane Doe\n"
        "Year Built: 1990\n"
        "Property Tax: 5000\n"
        "School Ratings: 8\n"
        "Neighborhood Desc: Spacious backyard with a swimming pool.\n"
        "Broker: XYZ Realty\n"
        "City: San Francisco\n"
        "State: CA\n"
        "Zip Code: 94118\n"
        "HOA Fees: 300\n\n"
        "**Trait:** Has a swimming pool.\n"
        "**Response:** yes\n\n"
        "#### Example 2:\n"
        "**Property Details:**\n"
        "Price: 600000\n"
        "Beds: 2\n"
        "Baths: 1\n"
        "Area: 1500 sqft\n"
        "Listing Agent: John Smith\n"
        "Year Built: 1985\n"
        "Property Tax: 4000\n"
        "School Ratings: 7\n"
        "Neighborhood Desc: Close to downtown parks.\n"
        "Broker: ABC Realty\n"
        "City: Irvine\n"
        "State: CA\n"
        "Zip Code: 92602\n"
        "HOA Fees: 250\n\n"
        "**Trait:** Includes a home gym.\n"
        "**Response:** no\n\n"
        "#### Example 3:\n"
        "**Property Details:**\n"
        "Price: 950000\n"
        "Beds: 4\n"
        "Baths: 3\n"
        "Area: 2500 sqft\n"
        "Listing Agent: Emily Clark\n"
        "Year Built: 2005\n"
        "Property Tax: 6000\n"
        "School Ratings: 9\n"
        "Neighborhood Desc: Recently renovated kitchen and hardwood floors.\n"
        "Broker: LMN Realty\n"
        "City: Redwood City\n"
        "State: CA\n"
        "Zip Code: 94061\n"
        "HOA Fees: 350\n\n"
        "**Trait:** Features hardwood floors.\n"
        "**Response:** yes\n\n"
        "#### Example 4:\n"
        "**Property Details:**\n"
        "Price: 720000\n"
        "Beds: 3\n"
        "Baths: 2\n"
        "Area: 1800 sqft\n"
        "Listing Agent: Michael Brown\n"
        "Year Built: 1995\n"
        "Property Tax: 4500\n"
        "School Ratings: 6\n"
        "Neighborhood Desc: Modern kitchen appliances.\n"
        "Broker: OPQ Realty\n"
        "City: Bronx\n"
        "State: NY\n"
        "Zip Code: 10451\n"
        "HOA Fees: 200\n\n"
        "**Trait:** Has a fireplace.\n"
        "**Response:** unsure\n\n"
        "#### Example 5 (Approximate City Match):\n"
        "**Property Details:**\n"
        "Price: 800000\n"
        "Beds: 3\n"
        "Baths: 2\n"
        "Area: 2200 sqft\n"
        "Listing Agent: Sarah Lee\n"
        "Year Built: 1998\n"
        "Property Tax: 5500\n"
        "School Ratings: 7\n"
        "Neighborhood Desc: Beautiful garden and modern kitchen.\n"
        "Broker: DEF Realty\n"
        "City: Redwood\n"
        "State: CA\n"
        "Zip Code: 94061\n"
        "HOA Fees: 320\n\n"
        "**Trait:** Located in Redwood City.\n"
        "**Response:** yes\n\n"
        "#### Example 6 (Misspelled City):\n"
        "**Property Details:**\n"
        "Price: 950000\n"
        "Beds: 4\n"
        "Baths: 3\n"
        "Area: 2600 sqft\n"
        "Listing Agent: Tom Hanks\n"
        "Year Built: 2000\n"
        "Property Tax: 6200\n"
        "School Ratings: 8\n"
        "Neighborhood Desc: Spacious living areas with hardwood floors.\n"
        "Broker: GHI Realty\n"
        "City: San Franciscso\n"
        "State: CA\n"
        "Zip Code: 94118\n"
        "HOA Fees: 400\n\n"
        "**Trait:** Located in San Francisco.\n"
        "**Response:** yes\n\n"
        "---\n\n"
        "### Now, evaluate the following:\n\n"
        "**Property Details:**\n"
        f"{property_details}\n\n"
        "**Trait:** {trait}\n"
        "**Response:**"
    )

    # Debug: Print the prompt (optional, remove in production)
    # print("Prompt sent to LLM:\n", prompt)

    # Get the response from OpenAI
    response = get_cached_response(
        prompt,
        max_tokens=3,
        temperature=0.0,
        stop=["\n"]  # Stop at the end of the line to prevent additional text
    ).strip().lower()
    print("Trait is :", trait)
    # Debug: Print the raw response (optional, remove in production)
    print("Raw LLM response:", response)

    # Map the response to the desired output
    if response == 'yes':
        return 'yes'
    elif response == 'no':
        return 'no'
    else:
        return 'unsure'



# ============================================
# Load and Preprocess Zillow Data
# ============================================
@st.cache_data
def load_data(file_path='Zillow_Data.csv'):
    if not os.path.exists(file_path):
        st.error(f"CSV file not found at path: {file_path}")
        return None
    df = pd.read_csv(file_path)
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(',', ''), errors='coerce')
    else:
        st.warning("'price' column not found in the dataset.")

    # Convert 'beds' and 'baths' to numeric
    df['beds'] = pd.to_numeric(df['beds'], errors='coerce')
    df['baths'] = pd.to_numeric(df['baths'], errors='coerce')
    # Convert 'city' and 'state' to title and upper case for consistent matching
    df['city'] = df['city'].str.title()
    df['state'] = df['state'].str.upper()
    # Ensure 'zip_code' is string
    if 'zip_code' in df.columns:
        df['zip_code'] = df['zip_code'].astype(str)
    else:
        st.warning("The dataset does not contain a 'zip_code' column.")
    return df


# List of stopwords to remove from traits (e.g., "has", "a", "the", "is near")
# List of phrases and words to remove from traits
removal_phrases = ['has', 'a', 'the', 'an', 'is near']


def extract_feature_from_trait(trait):
    """
    Extracts the relevant feature from the trait string by removing stopwords and matching 
    remaining words against the features_list.
    """
    # Tokenize the trait and remove unwanted words/phrases
    for phrase in removal_phrases:
        trait = trait.replace(phrase, '').strip()
    
    # Tokenize and clean up the trait
    tokens = [word.lower() for word in re.split(r'\W+', trait) if word]

    # Join tokens back to form the cleaned trait
    cleaned_trait = ' '.join(tokens)

    # Return the cleaned trait
    return cleaned_trait


def get_unique_cities(df):
    """
    Extract a sorted list of unique cities from the DataFrame.
    """
    return sorted(df['city'].dropna().unique())


# ============================================
# Load and Preprocess Broker Data
# ============================================
@st.cache_data
def load_broker_data(file_path='broker_data.csv'):
    if not os.path.exists(file_path):
        st.error(f"CSV file not found at path: {file_path}")
        return None
    df = pd.read_csv(file_path)
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()
    # Convert 'zip_code' to string to preserve leading zeros
    df['zip_code'] = df['zip_code'].astype(str)
    # Convert 'city' and 'state' to title and upper case for consistent matching
    df['city'] = df['city'].str.title()
    df['state'] = df['state'].str.upper()
    return df





# ============================================
# Main Function for Streamlit App
# ============================================
def main():
    st.title("🏠 Real Estate Query App")

    # Initialize session state
    initialize_session_state()

    # Initialize LLM cache
    initialize_llm_cache()

    # Add the sidebar for Previous Searches
    with st.sidebar:
        st.header("🔍 Previous Searches")
        if st.session_state.previous_searches:
            for idx, search in enumerate(st.session_state.previous_searches):
                # Truncate the search query for display
                truncated_query = truncate_text(search['query'], max_length=30)
                
                # Create a button with the search icon and truncated query
                button_label = f"🔍 {truncated_query}"
                # Assign a unique key using the index
                if st.button(button_label, key=f"prev_search_{idx}"):
                    # Load the selected previous search into session state
                    st.session_state.query = search['query']
                    st.session_state.user_intent = search['user_intent']
                    st.session_state.traits = search['traits']
                    st.session_state.key_phrases = search['key_phrases']
                    st.session_state.property_keywords = search['property_keywords']
                    st.session_state.sql_query = search['sql_query']
                    st.session_state.result = search['result']
                    st.session_state.page = 'page2'  # Navigate to the details page
                    st.rerun()
        else:
            st.write("No previous searches found.")

    # Check if client and data are loaded
    if st.session_state.client is None or st.session_state.df is None or st.session_state.broker_df is None:
        st.stop()

    # Page 1: User Input
    if st.session_state.page == 'page1':
        st.header("📥 Page 1: User Input")
        query = st.text_input("Enter your real estate query (e.g., '2 bed 2 bath in Irvine'):", key='user_query')
        if st.button("Submit"):
            if not query.strip():
                st.error("Please enter a valid query.")
            else:
                # Preprocess the query to fix common typos
                preprocessed_query = query
                st.session_state.query = preprocessed_query

                # Extract user intent, traits, and key phrases
                with st.spinner("Extracting information..."):
                    user_intent, traits, key_phrases = extract_information(preprocessed_query)
                st.session_state.user_intent = user_intent
                st.session_state.traits = traits
                st.session_state.key_phrases = key_phrases

                # Generate SQL query with retries
                with st.spinner("Generating SQL query..."):
                    sql_query = generate_sql_openai(
                        preprocessed_query, 
                        user_intent, 
                        traits, 
                        key_phrases, 
                        st.session_state.unique_cities, 
                        max_retries=5
                    )
                    st.session_state.sql_query = sql_query

                    if sql_query:
                        # Generate PropertyKeywords
                        with st.spinner("Generating Property Keywords..."):
                            property_keywords = get_property_keywords_openai(
                                preprocessed_query,
                                user_intent,
                                traits,
                                key_phrases,
                                sql_query
                            )
                        st.session_state.property_keywords = property_keywords

                        # Execute SQL query
                        with st.spinner("Executing SQL query..."):
                            result = execute_sql_query(sql_query, st.session_state.df)
                        st.session_state.result = result

                        # Save the current search to previous searches
                        new_search = {
                            'query': preprocessed_query,
                            'user_intent': user_intent,
                            'traits': traits,
                            'key_phrases': key_phrases,
                            'property_keywords': property_keywords,
                            'sql_query': sql_query,
                            'result': result
                        }
                        st.session_state.previous_searches.append(new_search)

                        # Optional: Keep only the last 10 searches to limit memory usage
                        if len(st.session_state.previous_searches) > 10:
                            st.session_state.previous_searches.pop(0)

                        # Navigate to Page 2
                        st.session_state.page = 'page2'
                        st.rerun()

                    else:
                        # Handle failure after retries
                        st.error("Could not generate SQL query after multiple attempts. Please try a different query or check your input.")

    # Page 2: Details
    elif st.session_state.page == 'page2':
        st.header("📝 Page 2: Query Details")

        # User Intent
        st.subheader("🔍 User Intent:")
        user_intent = st.session_state.get('user_intent', 'N/A')
        if isinstance(user_intent, str) and user_intent.strip():
            st.write(user_intent)
        else:
            st.write("No user intent information available.")

        # Traits
        st.subheader("📊 Traits:")
        traits = st.session_state.get('traits', [])
        if traits:
            for trait in traits:
                st.write(f"- {trait}")
        else:
            st.write("No traits information available.")

        # Key Phrases
        st.subheader("🗝️ Key Phrases:")
        key_phrases = st.session_state.get('key_phrases', [])
        if key_phrases:
            for phrase in key_phrases[:10]: 
                st.write(f"- {phrase}")
        else:
            st.write("No key phrases available.")

        # **New Section: PropertyKeywords**
        st.subheader("🗝️ Property Keywords:")
        property_keywords = st.session_state.get('property_keywords', '')
        if property_keywords:
            st.write(property_keywords)
        else:
            st.write("No Property Keywords available.")

        # Generated SQL Query
        st.subheader("💻 Generated SQL Query:")
        sql_query = st.session_state.get('sql_query', 'N/A')
        if sql_query:
            # Format SQL using sqlparse
            formatted_sql = sqlparse.format(sql_query, reindent=True, keyword_case='upper')
            st.code(formatted_sql, language='sql')
        else:
            st.write("No SQL query generated.")

        # Navigation buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔙 Back to Input"):
                go_to_page1()
        with col2:
            if st.button("➡️ Next to Results"):
                go_to_page3()

    # Page 3: Query Results
    elif st.session_state.page == 'page3':
        st.header("📈 Page 3: Query Results")
        
        # Add custom CSS to create a toggle switch
        st.markdown("""
            <style>
            /* The switch - the box around the slider */
            .switch {
                position: relative;
                display: inline-block;
                width: 60px;
                height: 34px;
            }

            /* Hide default HTML checkbox */
            .switch input {
                opacity: 0;
                width: 0;
                height: 0;
            }

            /* The slider */
            .slider {
                position: absolute;
                cursor: pointer;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: #ccc;
                transition: .4s;
                border-radius: 34px;
            }

            .slider:before {
                position: absolute;
                content: "";
                height: 26px;
                width: 26px;
                left: 4px;
                bottom: 4px;
                background-color: white;
                transition: .4s;
                border-radius: 50%;
            }

            input:checked + .slider {
                background-color: #2196F3;
            }

            input:checked + .slider:before {
                transform: translateX(26px);
            }
            </style>
            """, unsafe_allow_html=True)

        # Create the toggle button and a label for it
        st.markdown("""
            <label for="toggle-switch">Notify Property Alerts: </label>
            <label class="switch">
                <input type="checkbox" id="property-alert-toggle">
                <span class="slider"></span>
            </label>
            <script>
                const toggle = document.getElementById("property-alert-toggle");
                toggle.addEventListener("change", function() {
                    if (toggle.checked) {
                        alert("🔔 Property alerts are enabled.");
                    } else {
                        alert("🔕 Property alerts are disabled.");
                    }
                });
            </script>
        """, unsafe_allow_html=True)
        
        result = st.session_state.get('result')

        if result is not None and not result.empty:
            st.write(f"### 🏡 {len(result)} Properties Found:")

            st.write("### 🏡 Properties:")
            result = result.reset_index(drop=True)
            
            # Ensure 'beds' and 'baths' are integers
            if isinstance(result['beds'], pd.Series):
                result['beds'] = result['beds'].fillna(0).astype(int)
            if isinstance(result['baths'], pd.Series):
                result['baths'] = result['baths'].fillna(0).astype(int)
            
            # Extract traits from session state
            traits = st.session_state.get('traits', [])
            if not traits:
                st.warning("No traits available to generate dynamic columns.")
                dynamic_columns = []
            else:
                # Generate dynamic column names using OpenAI
                dynamic_columns = []
                trait_to_column = {}
                for trait in traits:
                    # Skip traits related to 'City'
                    # if 'city' in trait.lower():
                    #     continue
                    column_name = get_column_name_from_trait(trait)
                    # Ensure unique column names
                    original_column_name = column_name
                    counter = 1
                    while column_name in dynamic_columns:
                        column_name = f"{original_column_name}_{counter}"
                        counter += 1
                    dynamic_columns.append(column_name)
                    trait_to_column[trait] = column_name

                # Initialize dynamic columns with default white dots
                for column in dynamic_columns:
                    result[column] = "⚪"

                # Iterate over each trait and update the dynamic columns
                for trait in traits:
                    # Skip traits related to 'City'
                    # if 'city' in trait.lower():
                    #     continue
                    column_name = trait_to_column[trait]
                    st.spinner(f"Checking trait '{trait}' for properties...")
                    for idx, row in result.iterrows():
                        match_status = is_trait_matched(row, trait)
                        if match_status == 'yes':
                            result.at[idx, column_name] = "🟢"
                        elif match_status == 'no':
                            result.at[idx, column_name] = "⚪"
                        elif match_status == 'unsure':
                            result.at[idx, column_name] = "🟡"

            # Prepare display DataFrame
            display_columns = ['Property Link', 'Price'] + dynamic_columns + ['Description', 'Broker', 'Receive Realtor Proposal']

            display_df = pd.DataFrame()

            # Property Link
            display_df['Property Link'] = result['listingurl'].apply(
                lambda url: f"[View Property]({url})" if pd.notna(url) else "N/A"
            )

            # Price
            display_df['Price'] = result['price'].apply(
                lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
            )

            # Dynamic Columns (traits-based)
            for column in dynamic_columns:
                display_df[column] = result[column]

            # Description
            display_df['Description'] = result['neighborhood_desc'].apply(
                lambda desc: f"<div style='height: 100px; overflow: auto;'>{desc}</div>" if isinstance(desc, str) else "N/A"
            )

            # Broker (Clickable Link to Page 4)
            display_df['Broker'] = result['zip_code'].apply(
                lambda zip_code: "Click to view brokers" if pd.notna(zip_code) else "N/A"
            )

            # Receive Realtor Proposal (Button Placeholder)
            display_df['Receive Realtor Proposal'] = "Button Placeholder"

            # ============================================
            # 6. Display the DataFrame in Streamlit
            # ============================================
            # Display the header row with column names
            with st.container():
                cols = st.columns(len(display_columns))
                for col, header in zip(cols, display_columns):
                    col.markdown(f"**{header}**")

            # Display each property row
            for index, row in display_df.iterrows():
                with st.container():
                    cols = st.columns(len(display_columns))
                    # Property Link
                    cols[0].markdown(row['Property Link'], unsafe_allow_html=True)
                    # Price
                    cols[1].markdown(row['Price'], unsafe_allow_html=True)
                    # Dynamic Columns (traits-based)
                    for i, column in enumerate(dynamic_columns, start=2):
                        cols[i].markdown(row[column], unsafe_allow_html=True)
                    # Description
                    cols[-3].markdown(row['Description'], unsafe_allow_html=True)
                    # Broker Link
                    with cols[-2]:
                        if st.button("🔗 Click to view brokers", key=f"view_brokers_{index}"):
                            selected_zip = result.loc[index, 'zip_code']
                            st.session_state.selected_zip_code = selected_zip
                            st.session_state.page = 'page4'
                            st.rerun()
                    # Receive Realtor Proposal Button
                    with cols[-1]:
                        if st.button("📨 Receive Realtor Proposal", key=f"realtor_proposal_{index}"):
                            # Define the button's action here
                            st.success("Realtor proposal has been sent!")
                            # You can also add more complex logic, such as sending an email or saving to a database

            # Optionally, offer to download the results as CSV
            csv = result.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name='query_results.csv',
                mime='text/csv',
            )
        else:
            st.write("### ❌ No results found for the given query.")

        # Display the number of properties found
        if result is not None:
            st.write(f"### 🏠 Total Properties Found: {len(result)}")
        else:
            st.write("### 🏠 Total Properties Found: 0")

        # Navigation buttons
        st.write("---")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔙 Back to Details"):
                go_to_page2()
        with col2:
            if st.button("💾 Save Results to Text File"):
                save_to_txt(
                    file_name="query_output.txt",
                    query=st.session_state.get('query', ''),
                    user_intent=st.session_state.get('user_intent', ''),
                    traits=st.session_state.get('traits', []),
                    key_phrases=st.session_state.get('key_phrases', []),
                    property_keywords=st.session_state.get('property_keywords', ''),
                    sql_query=st.session_state.get('sql_query', ''),
                    final_output=result
                )
                st.success("Results saved to 'query_output.txt'.")






    # Page 4: Broker Details
    elif st.session_state.page == 'page4':
        st.header("📄 Page 4: Broker Details")

        selected_zip_code = st.session_state.get('selected_zip_code', None)

        if selected_zip_code:
            broker_df = st.session_state.get('broker_df', None)
            if broker_df is not None:
                # Filter brokers by the selected zip code
                filtered_brokers = broker_df[broker_df['zip_code'] == selected_zip_code]
                if not filtered_brokers.empty:
                    st.subheader(f"🏢 Brokers in Zip Code: {selected_zip_code}")
                    
                    # Select and rename columns for better readability
                    display_brokers = filtered_brokers.rename(columns={
                        'broker': 'Broker Name',
                        'zip_code': 'Zip Code',
                        'city': 'City',
                        'state': 'State',
                        'reviews': 'Reviews',
                        'recent_homes_sold': 'Recent Homes Sold',
                        'negotiations_done': 'Negotiations Done',
                        'years_of_experience': 'Years of Experience',
                        'rating': 'Rating'
                    })[['Broker Name', 'City', 'State', 'Zip Code', 'Reviews', 'Recent Homes Sold', 'Negotiations Done', 'Years of Experience', 'Rating']]

                    # Display the brokers in a table
                    st.dataframe(display_brokers.style.format({
                        'Rating': "{:.2f}"
                    }))
                else:
                    st.write(f"No brokers found for zip code: {selected_zip_code}")
            else:
                st.write("Broker data is not available.")
        else:
            st.write("No zip code selected.")

        # Navigation buttons
        st.write("---")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔙 Back to Results"):
                go_to_page3()
        with col2:
            if st.button("🏠 Back to Input"):
                go_to_page1()

    # Default case: reset to Page 1
    else:
        st.session_state.page = 'page1'
        st.rerun()


if __name__ == '__main__':
    main()




