from django.shortcuts import render
from django.http import JsonResponse
from .chatbot import Chatbot
from django.views.decorators.csrf import csrf_exempt

# Initialize the chatbot instance
chatbot_instance = Chatbot()

def chatbot_views(request):
    print("--" * 10)
    print("Request method:", request.method)
    #print("Request POST data:", request.POST)

    
    if request.method == 'POST':
        try:
            # get the user query and user role from the request
            user_query = request.POST.get('user-input', '')
            user_role = request.POST.get('user-role', '')
            print("user_query: ", user_query)
            print("user_role: ", user_role)
            
            response = chatbot_instance.answer_query(user_query,user_role)
            print("Response from chatbot:", response) 
            return JsonResponse({'response': response})
        
        
        except Exception as e:
            print("Exception occurred:", str(e))
            return JsonResponse({'error': str(e)}, status=500)
    else:
        print("Rendering index.html")
        return render(request, 'index.html')