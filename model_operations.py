# takes in a tensorflow/keras model 
# and input_review that is taken from user/tests 
# returns predicton
## NOTE: Check if the values are correct
def predict_from_model(model, input_review, length_input=100):
  pred = model.predict(input_review.reshape(1,length_input))
  return 'positive' if pred.argmax() == 1 else 'negative' if pred.argmax() == 2 else 'neutral'