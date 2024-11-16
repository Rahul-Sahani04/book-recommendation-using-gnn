import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from undetected_chromedriver import Chrome, ChromeOptions
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Set up ChromeOptions for headless browsing
chrome_options = ChromeOptions()
chrome_options.headless = True

user_data_dir = "/Users/rsahani/Library/Application Support/Google/Chrome/chrome_whatsapp_profile"
chrome_options.add_argument(f"--user-data-dir={user_data_dir}")
chrome_options.add_argument("--ignore-certificate-errors")
chrome_options.add_argument("--allow-running-insecure-content")

# Function to initialize the WebDriver
def init_driver():
    return Chrome(options=chrome_options)

# Function to scrape genres for a given ISBN
def scrape_genres(isbn):
    driver = init_driver()
    driver.get('https://www.goodreads.com/books/010231')
    try:
        search_box = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, 'q')))
        search_box.clear()
        search_box.send_keys(isbn)
        search_box.send_keys(Keys.RETURN)
        
        WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="__next"]/div[2]/main/div[1]/div[2]/div[2]/div[2]/div[5]/ul')))
        genres = driver.find_elements(By.XPATH, '//*[@id="__next"]/div[2]/main/div[1]/div[2]/div[2]/div[2]/div[5]/ul')
        genre_list = [genre.text for genre in genres]
        
        genre_list = genre_list[0].replace('\n', ',')
        
        return clean_genres(genre_list)
    except Exception as e:
        print(f"Error scraping ISBN {isbn}: {e}")
        return None
    finally:
        driver.quit()

def clean_genres(genre):
    genre = genre.split('\n')
    genre = [g.lower() for g in genre if g.lower() not in ['genres', '...more']]
    return ', '.join(genre)

# Read the CSV file
books_df = pd.read_csv('data/books.csv', low_memory=False)

# Function to process each ISBN and return the result
def process_isbn(isbn):
    return isbn, scrape_genres(isbn)

# Use ThreadPoolExecutor to run the scraper in parallel
batch_size = 1000  # Adjust the batch size as needed
for start in range(0, len(books_df), batch_size):
    end = min(start + batch_size, len(books_df))
    batch_isbns = books_df['ISBN'][start:end]
    
    print(f"Processing ISBNs {start} to {end}...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_isbn, isbn) for isbn in batch_isbns]
        results = []
        for future in as_completed(futures):
            results.append(future.result())
            
    
    print(f"Processed {len(results)} ISBNs.")
    
    # Update the DataFrame with the scraped genres
    for isbn, genres in results:
        books_df.loc[books_df['ISBN'] == isbn, 'Genres'] = genres
    
    print("Saving the updated DataFrame...")
    # Save the updated DataFrame to a new CSV file after each batch
    books_df.to_csv('data/books_with_genres.csv', index=False)