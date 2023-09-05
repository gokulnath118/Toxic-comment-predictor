import streamlit as st
import pickle
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

text=tf.keras.models.load_model("Toxic.h5")
with open('tokenizer', 'rb') as file:
    tokenizer = pickle.load(file)
#rain_sequences = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
# defining the function which will make the prediction using 
# the data which the user inputs
def predict(sentence):
    sequences = tokenizer.texts_to_sequences([sentence])
    padded_sequences = pad_sequences(sequences, maxlen=300, padding='post', truncating='post')
    # get predictions for toxicity
    predictions = text.predict(padded_sequences)
    if text.predict(padded_sequences)[0]>0.5:
        return 'Bad sequence'
    else:
        return 'Normal sequence'
# this is the main function in which we define our webpage 
def main():
    # giving the webpage a title
    st.title("Toxic Comment Classification")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:green;padding:13px">
    <h1 style ="color:black;text-align:center;">LET US FIND HOW TOXIC IS YOUR COMMENT</h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    comment= st.text_input("Type a comment")
    result=""
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
   # sample()
    if st.button("Predict"):
        result = predict(comment)
    st.success(result)
     
if __name__=='__main__':
    main()