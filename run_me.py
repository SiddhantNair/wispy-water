from __future__ import print_function, unicode_literals
from PyInquirer import prompt
from train_nn import evaluate_stuff, train_nn_model
from webcam import open_computer_vision

def menu_prompt():
    questions = [
        {
            'type': 'list',
            'name': 'choice',
            'message': 'What do you want to do?',
            'choices': [
                'Open Webcam',
                'Train Model',
                'Find Homer!',
                'Exit'
            ]
        }
    ]

    answers = prompt(questions)
    if answers['choice'] == 'Open Webcam':
        open_computer_vision()
    elif answers['choice'] == 'Exit':
        print('closing program... [Thank You]')
    elif answers['choice'] == 'Train Model':
        train_nn_model
    elif answers['choice'] == 'Find Homer!':
        evaluate_stuff()

if __name__ == "__main__":
    menu_prompt()