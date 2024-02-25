import streamlit as st
# Set the title using StreamLit
st.title(' Geeks For Geeks ')
input_text = st.text_input('Enter Your Text: ') 

if input_text: 
    title = chainT.run(input_text)
    wikipedia_research = wikipedia.run(input_text) 
    script = chainS.run(title=title, wikipedia_research=wikipedia_research)
 
    st.write(title) 
    st.write(script)