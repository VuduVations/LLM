!pip install google-cloud-aiplatform --upgrade
!pip install gradio
!pip install vertexai

!pip uninstall shapely
!pip install shapely==1.8.5

import vertexai
from vertexai.preview.language_models import ChatModel, InputOutputTextPair
import gradio as gr
vertexai.init(project="bionic-water-394610", location="us-central1")
from google.colab import auth as google_auth
google_auth.authenticate_user()


chat_model = ChatModel.from_pretrained("chat-bison@001")

parameters = {
  "temperature": 0.3,
  "max_output_tokens": 512,
  "top_p": 0.5,
  "top_k": 40
}

chat = chat_model.start_chat(
    context="""
    You are a customer service agent for Vegas Speedway""",

    examples = [
    InputOutputTextPair(input_text="When do you open?", output_text="For Exotics Racing, we are open 7 days a week, 9am-5pm. For Vegas Superkarts, we are open 7 days a week 10am-5pm."),
    InputOutputTextPair(input_text="How can I reach you?", output_text="You may reach us at (702) 802-5651 or you can e-mail us. We are here to assist you!"),
    InputOutputTextPair(input_text="Do you offer transportation to your track?", output_text="We do not offer transportation. We recommend taking an Uber, Lyft, or taxi to our location. Prices start at $40-$60 one-way. Alternatively, Presidential Limo by KAPTYN offers private transportation to EXR. Please call 702-438-5466 to book."),
    InputOutputTextPair(input_text="How should I dress for driving?", output_text="Drivers need to wear comfortable clothes and closed-toe shoes, preferably sneakers. Helmets are mandatory and provided free of charge. Sandals and high-heels are not allowed for drivers."),
    InputOutputTextPair(input_text="Are cameras allowed in the cars?", output_text="Because of our insurance policy, we do not allow any other camera or video equipment in the cars. However, we do offer an on-board video of your driving experience that you may purchase prior to driving."),
    InputOutputTextPair(input_text="How long should I plan to be at the track?", output_text="Plan for a minimum of 2 hours. If you are driving for more than 50 laps, plan for at least 6 hours from start to finish."),
    InputOutputTextPair(input_text="Can I change which car I drive once I'm there?", output_text="Yes, depending on availability, you may be able to change which car you drive or drive additional cars. You can also buy additional laps or choose add-ons such as a video of your driving experience."),
    InputOutputTextPair(input_text="Do you have any combo packages?", output_text="Yes, we offer the ability to book combo packages. These combos are designed for one driver and not to be split up with multiple drivers. The combo package helps provide you with the opportunity to compare and contrast power and handling of these exciting supercars."),
    InputOutputTextPair(input_text="Are snacks available at your facility?", output_text="Yes. Snacks and beverages are available for purchase. However, please note that alcohol is not permitted at our track facility."),
    InputOutputTextPair(input_text="What happens if it rains?", output_text="All our cars are equipped with security systems and tires to handle any condition. In extreme cases where the weather forces us to close the track, we can reschedule you at any of our tracks, free of charge. However, some specific cars like Ferrari 488 Challlenge Evo, McLaren 570 GT4, and others may not be able to run in the rain."),
    InputOutputTextPair(input_text="Can multiple drivers share a car package?", output_text="All our packages and multiple car packages are per driver and cannot be split amongst multiple drivers."),
    InputOutputTextPair(input_text="How can I buy a Gift Certificate?", output_text="You can purchase a Gift Certificate online at our website http://www.vegassupercars.com. Or you can purchase one over the phone by calling us at (702) 802-5651."),
    InputOutputTextPair(input_text="What payment methods do you accept?", output_text="We accept various payment methods: 1. Credit Card: Visa, MasterCard, Discover, and American Express. 2. Check: Order is only confirmed after full payment. 3. Cash: Possible on the track, but we can't guarantee availability without prior payment."),
    InputOutputTextPair(input_text="Do I have to pay for the reservation upfront?", output_text="Yes, the reservation must be paid in full prior to your reservation date."),
    InputOutputTextPair(input_text="Is the Gift Certificate valid indefinitely?", output_text="No, our Gift Certificates are valid for a period of 1 year from the date of purchase. They are non-refundable and cannot be exchanged for cash."),
    InputOutputTextPair(input_text="Can children participate?", output_text="Children cannot drive the cars, but they can be passengers in the drift ride-along experience if they are at least 60lbs (27kg)."),
    InputOutputTextPair(input_text="How old do I need to be to drive?", output_text="Drivers must be at least 18 years old and hold a valid driving license from their country of residence."),
    InputOutputTextPair(input_text="Do I need to have a racing background?", output_text="No racing experience is necessary. We have instructors who will guide you through the process."),
    InputOutputTextPair(input_text="What if I damage a car?", output_text="We offer insurance for purchase to cover any damages that may occur during your experience. The price depends on the car model you choose to drive."),
    InputOutputTextPair(input_text="How many cars can I drive in one day?", output_text="You can drive as many cars as you'd like in one day. We have combo packages designed for this purpose."),
    InputOutputTextPair(input_text="Can I get a refund if I can't make it?", output_text="Refunds are not typically provided, but you may reschedule your experience within 48 hours of your original booking time."),
    InputOutputTextPair(input_text="Are there any health restrictions?", output_text="All participants must be in good physical health. We recommend consulting with your physician if you have any concerns."),
    InputOutputTextPair(input_text="Can I bring guests to watch?", output_text="Absolutely! We have a dedicated viewing area for your guests to watch and cheer you on."),
    InputOutputTextPair(input_text="Do you offer any discounts?", output_text="We occasionally have promotions and special offers. We recommend subscribing to our newsletter or checking our website for current deals."),
    InputOutputTextPair(input_text="Can I book a corporate event?", output_text="Yes! We offer corporate events and team-building experiences. Contact our sales team for more details."),
    InputOutputTextPair(input_text="Do I need to bring my own helmet?", output_text="No, we provide helmets free of charge. However, if you have a personal helmet that meets our safety standards, you are welcome to bring it."),
    InputOutputTextPair(input_text="How often do you maintain your cars?", output_text="Our cars undergo rigorous maintenance checks after every driving session. They're inspected daily to ensure they're in peak condition for our guests."),
    InputOutputTextPair(input_text="Do you have driving instructors available?", output_text="Yes, we have professional driving instructors on-site to guide and assist you throughout your driving experience."),
    InputOutputTextPair(input_text="Can I get a group discount?", output_text="Group discounts are available for groups of 10 or more. Please contact our sales team for specific pricing and details."),
    InputOutputTextPair(input_text="How long is the track?", output_text="Our track is 1.2 miles long, featuring a mix of challenging turns and high-speed straights."),
    InputOutputTextPair(input_text="Can I buy merchandise?", output_text="Yes, we have a store on-site where you can purchase apparel, souvenirs, and other merchandise related to our cars and brand."),
    InputOutputTextPair(input_text="Are there any weight or height limits?", output_text="While we don't have strict weight or height limits, certain cars might be uncomfortable for individuals above 6'5\" or those over 250 lbs. We recommend checking with our staff about the best car choices for your comfort."),
    InputOutputTextPair(input_text="Do I get a certificate after my drive?", output_text="Yes, after your driving experience, you'll receive a certificate commemorating your day at the track with us."),
    InputOutputTextPair(input_text="What's the fastest car you have?", output_text="Our fastest car in terms of top speed is the Bugatti Veyron, capable of speeds over 250 mph. However, different cars offer unique driving experiences, so it's worth trying several to find your favorite."),
    InputOutputTextPair(input_text="Do I need to bring my own helmet?", output_text="No, we provide helmets free of charge. However, if you have a personal helmet that meets our safety standards, you are welcome to bring it."),
    InputOutputTextPair(input_text="How often do you maintain your cars?", output_text="Our cars undergo rigorous maintenance checks after every driving session. They're inspected daily to ensure they're in peak condition for our guests."),
    InputOutputTextPair(input_text="Do you have driving instructors available?", output_text="Yes, we have professional driving instructors on-site to guide and assist you throughout your driving experience."),
    InputOutputTextPair(input_text="Can I get a group discount?", output_text="Group discounts are available for groups of 10 or more. Please contact our sales team for specific pricing and details."),
    InputOutputTextPair(input_text="How long is the track?", output_text="Our track is 1.2 miles long, featuring a mix of challenging turns and high-speed straights."),
    InputOutputTextPair(input_text="Can I buy merchandise?", output_text="Yes, we have a store on-site where you can purchase apparel, souvenirs, and other merchandise related to our cars and brand."),
    InputOutputTextPair(input_text="Are there any weight or height limits?", output_text="While we don't have strict weight or height limits, certain cars might be uncomfortable for individuals above 6'5\" or those over 250 lbs. We recommend checking with our staff about the best car choices for your comfort."),
    InputOutputTextPair(input_text="Are pets allowed at the track?", output_text="Sorry, pets are not allowed at our track facility for their safety and the safety of our guests."),
    InputOutputTextPair(input_text="""How much does the supercar VIP Experience cost?""", output_text="""The supercar VIP Experience is available from $1,000 for drivers who desire a one-of-a-kind experience."""),
    InputOutputTextPair(input_text="""Can I drive a Baja Race Truck at your facility?""", output_text="""Yes, you can experience the thrill of driving a Baja Race Truck on our 1 mile, 3 high jump off-road track with one of our highly trained Experience Specialists. The experience includes high-banked and high-speed corners, and jumps as high as 6 feet."""),
    InputOutputTextPair(input_text="""How much is the off-road driving experience?""", output_text="""The off-road driving experience starts at $399."""),
    InputOutputTextPair(input_text="""How many laps can I get for $299?""", output_text="""You can get 5 laps for $299."""),
    InputOutputTextPair(input_text="""How much is each lap?""", output_text="""Each lap is priced from $45."""),
    InputOutputTextPair(input_text="""What can I expect when I arrive?""", output_text="""Our friendly team will welcome you for an unforgettable experience. You'll learn the most efficient techniques to drive on our dirt-track, followed by 2 free discovery laps in a Ford Raptor."""),
    InputOutputTextPair(input_text="""Can I drive other off-road cars on your track?""", output_text="""Yes, you can get behind the wheel of powerful off-road cars on our track."""),
    InputOutputTextPair(input_text="""Can I add more cars or laps to my experience?""", output_text="""Yes, you can enhance your experience by adding more cars, laps, ride-along, and go-kart."""),
    InputOutputTextPair(input_text="""Can I purchase a video or photo of my experience?""", output_text="""Yes, you can treasure your experience forever by purchasing the On-Board HD video and customized photo."""),
    InputOutputTextPair(input_text="""Where is VEGAS OFF-ROAD EXPERIENCE located?""", output_text="""VEGAS OFF-ROAD EXPERIENCE is located only 10 minutes south of the strip, at the fabulous SPEEDVEGAS MOTORSPORT PARK."""),
    InputOutputTextPair(input_text="""Tell me more about the off-road Race Truck experience.""", output_text="""Experience the rush of driving a real Race Truck on our 1 mile off-road track with one of our professional Experience Specialists. You'll defy gravity on 3 high jumps that launch you up to 10 feet into the air, and experience adrenaline pumping through 8 heart-racing turns and high-banked corners."""),
    InputOutputTextPair(input_text="""What are your operating hours?""", output_text="""We are open 7 days a week from 9:00AM to 5:00PM."""),
    InputOutputTextPair(input_text="""Can I ride along in a supercar?""", output_text="""Yes, if you want to experience the speed of supercars like the 911 GT3 RS, Ferrari 488 GTB, or Lamborghini Huracan, we suggest going for Exotic Racing’s Supercar Ride-Along. You can be a co-pilot in a high-speed lap with our finest instructor behind the wheel."""),
    InputOutputTextPair(input_text="""How much is the Supercar Ride-Along?""", output_text="""The Supercar Ride-Along costs $149 for 3 laps."""),
    InputOutputTextPair(input_text="""How much horsepower does the quintessential supercar have?""", output_text="""The quintessential supercar has 580 hp sent to rear wheels."""),
    InputOutputTextPair(input_text="""What can I expect from the supercar experience?""", output_text="""The quintessential supercar is not just a prestigious piece of art on four wheels. It is an immensely capable track weapon. Hold tight and try not to scream when experiencing its power."""),
    InputOutputTextPair(input_text="""Can I jump with the Race Truck?""", output_text="""Yes, with the Race Truck you can defy gravity on 3 high jumps that can launch you up to 10 feet into the air."""),
    InputOutputTextPair(input_text="""Do I get to drive on my own?""", output_text="""You will be driving with one of our professional Experience Specialists to ensure safety and guidance throughout your off-road experience."""),
    InputOutputTextPair(input_text="""How high can the Baja Race Truck jump?""", output_text="""The Baja Race Truck can reach jumps as high as 6 feet."""),
    InputOutputTextPair(input_text="""Are there any special techniques I need to know for driving on the dirt track?""", output_text="""Yes, when you arrive you will learn the most efficient techniques to drive on our dirt-track."""),
    InputOutputTextPair(input_text="""Can I bring my own car?""", output_text="""The information provided doesn't specify if personal cars are allowed. It's best to contact VEGAS SUPERCARS directly."""),
    InputOutputTextPair(input_text="""Is there a difference between the $149 / 3 LAPS prices?""", output_text="""The prices listed are the same: $149 for 3 laps."""),
    InputOutputTextPair(input_text="""Do you offer any packages that combine different experiences?""", output_text="""Yes, we offer the ability to book combo packages to compare and contrast power and handling of different supercars."""),
    InputOutputTextPair(input_text="""Is there any vehicle that lets me experience high jumps?""", output_text="""Yes, the Baja Race Truck allows you to experience high jumps on our off-road track."""),
    InputOutputTextPair(input_text="""What can I expect from driving the Race Truck?""", output_text="""Experience the adrenaline pumping through 8 heart racing turns and high-banked corners, and defy gravity on 3 high jumps that can launch you up to 10 feet into the air."""),
    InputOutputTextPair(input_text="Which cars can I experience with Exotic Racing's Supercar Ride-Along?", output_text="You can experience the 911 GT3 RS, Ferrari 488 GTB, or Lamborghini Huracan with Exotic Racing’s Supercar Ride-Along."),
    InputOutputTextPair(input_text="What is the experience of the Supercar Ride-Along?", output_text="In the Supercar Ride-Along, you will be a co-pilot in the high-speed lap with our finest instructor behind the wheel. You will be amazed by both the speed and the skills of our drivers."),
    InputOutputTextPair(input_text="How much does the Supercar Ride-Along cost?", output_text="The Supercar Ride-Along costs $149 for 3 laps."),
    InputOutputTextPair(input_text="How many laps do I get for $149?", output_text="For $149, you get 3 laps in the Supercar Ride-Along."),
    InputOutputTextPair(input_text="Who will be driving during the Supercar Ride-Along?", output_text="During the Supercar Ride-Along, our finest instructor will be behind the wheel."),
    InputOutputTextPair(input_text="Can I be the driver in the Supercar Ride-Along?", output_text="No, in the Supercar Ride-Along you will be the co-pilot, with our instructor driving."),
    InputOutputTextPair(input_text="What will amaze me during the Supercar Ride-Along?", output_text="You will be amazed not just by the sheer speed but also by the incredible skills of our drivers."),
    InputOutputTextPair(input_text="Is there a difference between the $149 / 3 LAPS prices?", output_text="The prices listed are the same: $149 for 3 laps."),
    InputOutputTextPair(input_text="How fast can the cars go during the Supercar Ride-Along?", output_text="The cars can go very fast during the Supercar Ride-Along, allowing you to experience their top speeds and the skills of our drivers."),
    InputOutputTextPair(input_text="Who offers the Supercar Ride-Along experience?", output_text="The Supercar Ride-Along experience is offered by Exotic Racing."),
    InputOutputTextPair(input_text="Do I need to attend any briefing before driving?", output_text="Yes, every driver is required to attend the technical briefing where professional instructors explain various driving techniques and car specifications."),
    InputOutputTextPair(input_text="Do participants need to wear helmets?", output_text="Yes, every driver and rider must wear helmets while in the car and on the track. Vegas Supercars provides helmets at no charge for all participants."),
    InputOutputTextPair(input_text="Can my friends and family watch?", output_text="Yes, friends and family are welcome to watch from secure viewing areas. They can also take videos and pictures. However, pets are not allowed, only service animals."),
    InputOutputTextPair(input_text="Is there any age restriction for children attending the experience?", output_text="Children are welcome. However, for safety reasons, it's not recommended for children under the age of 24 months to attend."),
    InputOutputTextPair(input_text="Who is allowed in the car during the driving experience?", output_text="Only a driver and instructor will be in the car. Guests can accompany the participant for their discovery laps at a cost of $20 per additional person."),
    InputOutputTextPair(input_text="Can I bring my own camera or video equipment?", output_text="No, due to the insurance policy, personal camera or video equipment is not allowed in the cars. However, there's an on-board video service offered that you can purchase before driving."),
    InputOutputTextPair(input_text="How long should I plan for the driving experience?", output_text="You should plan for a minimum of 2 hours. However, if you are driving for more than 50 laps, you should plan for at least 6 hours from start to finish."),
    InputOutputTextPair(input_text="Can children or pets come along?", output_text="Children are welcome but it's not recommended for those under 24 months. Pets are not allowed, only service animals are permitted."),
    InputOutputTextPair(input_text="What will I learn during the technical briefing?", output_text="During the technical briefing, you'll learn about the correct driving position, steering techniques, car specifications, and correct racing lines."),
    InputOutputTextPair(input_text="Is there a charge for helmets?", output_text="No, Vegas Supercars offers helmets at no charge for all participants."),
    InputOutputTextPair(input_text="Do I need to attend any briefing before driving?", output_text="Yes, every driver is required to attend the technical briefing where professional instructors explain various driving techniques and car specifications."),
    InputOutputTextPair(input_text="Do participants need to wear helmets?", output_text="Yes, every driver and rider must wear helmets while in the car and on the track. Vegas Supercars provides helmets at no charge for all participants."),
    InputOutputTextPair(input_text="Can my friends and family watch?", output_text="Yes, friends and family are welcome to watch from secure viewing areas. They can also take videos and pictures. However, pets are not allowed, only service animals."),
    InputOutputTextPair(input_text="Is there any age restriction for children attending the experience?", output_text="Children are welcome. However, for safety reasons, it's not recommended for children under the age of 24 months to attend."),
    InputOutputTextPair(input_text="Who is allowed in the car during the driving experience?", output_text="Only a driver and instructor will be in the car. Guests can accompany the participant for their discovery laps at a cost of $20 per additional person."),
    InputOutputTextPair(input_text="Can I bring my own camera or video equipment?", output_text="No, due to the insurance policy, personal camera or video equipment is not allowed in the cars. However, there's an on-board video service offered that you can purchase before driving."),
    InputOutputTextPair(input_text="How long should I plan for the driving experience?", output_text="You should plan for a minimum of 2 hours. However, if you are driving for more than 50 laps, you should plan for at least 6 hours from start to finish."),
    InputOutputTextPair(input_text="Can children or pets come along?", output_text="Children are welcome but it's not recommended for those under 24 months. Pets are not allowed, only service animals are permitted."),
    InputOutputTextPair(input_text="What will I learn during the technical briefing?", output_text="During the technical briefing, you'll learn about the correct driving position, steering techniques, car specifications, and correct racing lines."),
    InputOutputTextPair(input_text="Is there a charge for helmets?", output_text="No, Vegas Supercars offers helmets at no charge for all participants.")


]

# Assuming the `InputOutputTextPair` class has been defined, you can iterate over this list and use the pairs as needed.

)

def chapel_chatbot(message, history):
    response = chat.send_message(message)
    return response.text

vspeed = gr.ChatInterface(
    fn=chapel_chatbot,
    title='SpeedVegas',
    description='Experience VoodooVations -advanced AI-powered Natural Language Processing. Seamlessly integrate with your existing website and witness the clear advantages firsthand.',
    examples = ['What packages do offer?', 'Do you offer transporation', 'How fast can you go?'],

  )

vspeed.launch(share=True)


from moviepy.editor import *

# Load video
video = VideoFileClip('/content/vodoovations_nlp2.mp4')

# Extract audio
audio = video.audio

# Save audio
audio.write_audiofile('output_audio.mp3')

# Close the audio object
audio.close()

