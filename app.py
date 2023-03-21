from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@app.route('/')
def my_form():
    return render_template('form.html', title='Sentiment Analysis')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text1']
    
    # Validate user input
    if not text or '<'in text:
        return render_template('form.html', title='Sentiment Analysis', error='Invalid input')
    
    labels = ['positive', 'negative', 'neutral','happy','sad','angry','sarcasm']
    results = classifier(text, labels)
    sentiment = results['labels'][0]
    print(sentiment)
    print(results)
    s1=results['labels'][0]
    s2=results['labels'][1]
    s3=results['labels'][2]
    s4=results['labels'][3]
    s5=results['labels'][4]
    s6=results['labels'][5]
    s7=results['labels'][6]
    score1 = round(results['scores'][0], 3)
    score2 = round(results['scores'][1], 3)
    score3 = round(results['scores'][2], 3)
    score4 = round(results['scores'][3], 3)
    score5 = round(results['scores'][4], 3)
    score6 = round(results['scores'][5], 3)    
    score7 = round(results['scores'][6], 3)   
    return render_template('form.html', title='Sentiment Analysis', text=text,final=sentiment,text1=score1, text2=score2, text3=score3, text4=score4,text5=score5,text6=score6,text7=score7,
                                                                                    sentiment1=s1,sentiment2=s2,sentiment3=s3,sentiment4=s4,sentiment5=s5,sentiment6=s6,sentiment7=s7)