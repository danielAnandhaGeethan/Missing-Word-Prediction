import streamlit as st
import subprocess

def main():
    
    st.markdown("<h1 style='text-align: center; color: #000000'>Missing Word Prediction</h1>", unsafe_allow_html = True)

    st.write("")
    text = st.text_input(label = 'Enter Your Sentence')

    btn = st.button('Get Prediction', use_container_width = True)

    res = ["","",""]
    if btn:
        if text == "":
            st.error("Enter a Sentence!!!")
        else:
            result = subprocess.run(['python', './prediction.py'] + [text], stdout=subprocess.PIPE, text=True)
            result = result.stdout.strip("'")

            res = []
            temp = ""
            for j,i in enumerate(result):
                if i == "[" or (i == " " and result[j-1] == ","):
                    pass
                elif (i == "," and result[j+1] == " ") or i == "]":
                    res.append(temp.strip("'"))
                    temp = ""
                else:
                    temp += i

    st.write("")
    st.write("")

    st.markdown(f'<h1 style = "font-size: 20px; color: #000000;">Most Possible Prediction : <span style="font-size: 20px; color: #2e6266">{res[0]}</span></h1>', unsafe_allow_html = True)

    st.write("")

    st.markdown(f'<h1 style = "font-size: 20px; color: #000000;">Sentence after Prediction : <span style="font-size: 20px; color: #510b3b">{res[1]}</span></h1>', unsafe_allow_html = True)

    st.write("")
    st.write("")

    if res[2] != "":
        res[2] = str(res[2]).split('\n')[0].split('\\n')

        temp = ""
        for i in res[2]:
            temp += i
            temp += "\n"
        
        res[2] = temp.strip("\n")

    st.text_area(label = 'Other Alternatives', value = res[2], height = 150)

if __name__ == "__main__":
    main()