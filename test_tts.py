from inora_tts import INORASpeaker, INORAMessages
import time

tts = INORASpeaker(lang='fr')
msg = INORAMessages

print('=== Test INORA TTS ===')
print('Commandes : urgent / ocr / monnaie / en / fr / quitter')
print('Ou tape directement un message')

while True:
    commande = input('\n> ')
    
    if commande == 'quitter':
        break
    elif commande == 'urgent':
        tts.say_urgent(msg.get('obstacle_near', 'fr', dist=80))
    elif commande == 'ocr':
        texte = input('Texte OCR à lire : ')
        tts.toggle()
        tts.say(msg.get('ocr_reading', 'fr', text=texte), priority='normal')
    elif commande == 'monnaie':
        tts.say(msg.get('coin_detected', 'fr', value='Pièce de deux euros'), priority='high')
    elif commande == 'en':
        tts.set_language('en')
        tts.say(msg.get('system_ready', 'en'), priority='normal')
    elif commande == 'fr':
        tts.set_language('fr')
        tts.say(msg.get('system_ready', 'fr'), priority='normal')
    else:
        tts.say(commande, priority='normal')
    
    time.sleep(0.5)

tts.stop()