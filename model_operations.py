"""
# takes in a tensorflow/keras model 
# and input_review that is taken from user/tests 
# returns predicton
## NOTE: Check if the values are correct
"""
def predict_from_model(model, input_review, length_input=100):
  pred = model.predict(input_review.reshape(1,length_input))
  max_index=0
  max_val = 0 
  current = 0
  for x in pred[0]:
    if max_val < x:
      max_val = x
      max_index = current
    current += 1
  return 'positive' if max_index == 2 else 'negative' if max_index == 0 else 'neutral'