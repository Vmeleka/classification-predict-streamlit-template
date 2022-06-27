"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	#st.title("Tweet Classifer")
	#st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "About the company"]
	selection = st.sidebar.selectbox("Choose Option", options)

    
    
    
    
    
	# Building out the "Information" page
	if selection == "Information":
		st.title("Project Information")
		st.subheader("Climate change insight")
		st.info(" To Provide an accurate and robust solution to companies to access a broad base of consumer sentiment, spanning multiple demographic and geographic categories hence increasing their insights and informing future marketing strategies.")
		# You can read a markdown file from supporting resources folder
		st.markdown("A model that is able to classify whether or not a person believes in climate change, based on their novel tweet data")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
            
            
            
            
            

	# Building out the predication page
	if selection == "Prediction":
		st.title("Tweet Classifier")     
		st.subheader("Climate change tweet classification")
		st.info("Prediction with ZF3 Machine learning Model")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter your Text","Type Here")        
        
        # you can create multiple pages this way
		options = ["lm", "naive", "mcv"]
		selection = st.sidebar.selectbox("Select Model", options)

		if st.button("Click to Classify"):
		# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
		# Load your .pkl file with the model of your choice + make predictions
		# Try loading in multiple models to give the user a choice
            
			if selection == "lm":  
				predictor = joblib.load(open(os.path.join("forest_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
                    
			if selection == "naive":
				predictor = joblib.load(open(os.path.join("forest_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
                    
			if selection == "mcv":  
				predictor = joblib.load(open(os.path.join("forest_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)                   
                   
                    
#				predictor = joblib.load(open(os.path.join("forest_model.pkl"),"rb"))
#				prediction = predictor.predict(vect_text)





	# Building our company's profile page
	if selection == "About the company":
		st.title("Company's profile") 
		st.subheader("Welcome to ZF3 company")
		st.info("The company was founded in June 2022 with the following pioneers")
		# You can read a markdown file from supporting resources folder
        
        

        
        #image doc code format

		from PIL import Image
		image1 = Image.open('abcf.jpg')        
		victor_image = Image.open('victor.jpg')
		Francis_image = Image.open('abcf.jpg')        
		Abigael_image = Image.open('abcf.jpg')
		Michael_image = Image.open('abcf.jpg')
		Raheeemat_image = Image.open('abcf.jpg')
		Harmony_image = Image.open('abcf.jpg') 
        
		st.image(image1, caption='Sunrise by the mountains', width=350)
		st.image(victor_image, caption='Model Expert', width=350) 
        
        #documentation for images
        #st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        
        



		st.markdown("Francis Ikegwu,Abigael Kinini, Michael Mamah, Raheeemat Adetunji,Victor Meleka, Harmony Odumuko") 

		st.subheader("More information")
		if st.checkbox('Show contact information'): # data is hidden if box is unchecked
			st.info("francisikegwu@yahoo.com, kininiabigael@gmail.com,mamahchidike@gmail.com, icontola@gmail.com, vicmeleka@gmail.com, nibotics@gmail.com") # will write the df to the page  
        # video format /doc
		video_file = open('sdw.mp4', 'rb')
		video_bytes = video_file.read()

		st.video(video_bytes)            
            

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			#st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
