import lmstudio as lms
import streamlit as st

st.set_page_config(page_title="🦜🔗 Quickstart App")
st.title('🦜🔗 Quickstart App')

def generate_response(input_text):
  result = model.respond(input_text)
  st.info(result(input_text))
  return result

with st.form('my_form'):
  text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
  submitted = st.form_submit_button('Submit')
  model = lms.llm("deepseek-r1-distill-qwen-14b")
  result = generate_response(text)
  st.write(result)