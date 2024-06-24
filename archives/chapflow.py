!pip install shapely==1.8.5
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
  "top_p": 0.8,
  "top_k": 40
}

chat = chat_model.start_chat(
    context="""
    You can refer people to sign up for the services at this link: https://www.littlechapel.com/reservations/get-started. You will need to reference these packages and prices: basic package is $299, the Tradtional package is $495, the romantic pack is $650, the Regal Package is $795
    the Exquisite package is $1099, the Luxurious package is $1399, the Cherish package is $2099, the Cara Mia package is $3199, the Vero Amore package is $6199, the Bella Elaganza Package is $9199,
    the intimate elopment package is $899, the lucky in love package is $999, the Viva Elvis package is $1099, the Eternal Love package is $1199,
    the high roller package is $1799,, the elegant grand canyon luncheon is $1999, the One in a Million package is $2022, the Valley of Fire package is $2399,
    the red rock canyon package is $3899, the majestic grand canyon package is $5399, the just the two of us $295, simple package $345, renewal of vows $599,
    tradtional ceremony and reception is $1899, regal ceremony and reception $4899, timeless ceremony and reception is $6299, the Forever Ceremony and reception is $8888.
    You can direct the cusotmer to https://www.littlechapel.com/wedding-packages for further clarifiction on the packages.  You should always suggest the highest package price and package
    when the customer provides a budget""",
    examples=[

        InputOutputTextPair(
            input_text="""What is the base price of the Elegant Grand Canyon Luncheon package?""",
            output_text="""The base price of the Elegant Grand Canyon Luncheon package is $1,999."""
        ),
        InputOutputTextPair(
            input_text="""What is the base pricing on the High Roller package?""",
            output_text="""The base price of the High Roller package is $1,799."""
        ),
        InputOutputTextPair(
            input_text="What is the base price for the Regal Ceremony and Reception package?",
            output_text="The base price of the Regal Ceremony and Reception package is $4,899."
        ),
        InputOutputTextPair(
            input_text="""How can I customize my Regal bouquet?""",
            output_text="""The Regal package includes an 18 rose bridal bouquet but special colors and upgrades can be requested to customize it."""
        ),
        InputOutputTextPair(
            input_text="What photography is in the Regal package?",
            output_text="The Regal package includes an 11x14 print, a 20 minute on-site photo shoot, and 4x6 candid prints."
        ),
        InputOutputTextPair(
            input_text="""What cake is included in the Regal Ceremony and Reception package?""",
            output_text="""The Regal Ceremony and Reception package includes a 2-tier cake serving 18-25 guests."""
        ),
        InputOutputTextPair(
            input_text="What is the pricing for the Bella Eleganza package?",
            output_text="The base price of the Bella Eleganza package is $9,199."
        ),
        InputOutputTextPair(
            input_text="""Can I have an additional limousine for my fiancé with the Bella Eleganza package?""",
            output_text="""Yes, an additional luxurious limousine can be added for your fiancé, allowing both of you to arrive separately to the chapel."""
        ),
        InputOutputTextPair(
            input_text="""How can I relive the special moments from my Bella Eleganza wedding?""",
            output_text="""Your ceremony will be available on a thumb drive. There's also an option for 16 weeks of internet viewing to rewatch your ceremony."""
        ),
        InputOutputTextPair(
            input_text="""What dining options are available post-wedding in the Bella Eleganza package?""",
            output_text="""The package offers a romantic dinner for two at various top-notch partnered restaurants with multiple menu choices."""
        ),
        InputOutputTextPair(
            input_text="""Is there a cake included in the Bella Eleganza package?""",
            output_text="""Yes, the package includes a 1-tier wedding cake serving 15-18 people, with customization options. The cake can be delivered to the chapel, hotel, or reception venue."""
        ),
        InputOutputTextPair(
            input_text="""What kind of floral décor is included in the Bella Eleganza package?""",
            output_text="""The package includes extravagant custom pew markers, a one-of-a-kind bridal bouquet, bridesmaid bouquet, a large altar spray, silk rose petals for the aisle, and custom boutonniere designs."""
        ),
        InputOutputTextPair(
            input_text="What is the base price point for Cara Mia?",
            output_text="The base price of the Cara Mia package is $3,199."
        ),
        InputOutputTextPair(
            input_text="What is the pricing on the Cherish package?",
            output_text="The base price of the Cherish package is $2,099."
        ),
        InputOutputTextPair(
            input_text="What is the base price of the Exquisite package?",
            output_text="The base price of the Exquisite package is $1,099."
        ),
        InputOutputTextPair(
            input_text="What is the pricing for the Intimate Elopement package?",
            output_text="The base price of the Intimate Elopement package is $899."
        ),
        InputOutputTextPair(
            input_text="What is the base price for the Viva Elvis package?",
            output_text="$1,099"
        ),
        InputOutputTextPair(
            input_text="What is the base price of the Valley of Fire package?",
            output_text="The base price of the Valley of Fire package is $2,399."
        ),
        InputOutputTextPair(
            input_text="What is the price of the Renewal of Vows package?",
            output_text="The base price of the Renewal of Vows package is $599."
        ),
        InputOutputTextPair(
            input_text="What is the base price for the Majestic Grand Canyon package?",
            output_text="The base price of the Majestic Grand Canyon package is $5,399."
        ),
        InputOutputTextPair(
            input_text="What is the price point of the Luxurious package?",
            output_text="The base price of the Luxurious package is $1,399."
        ),
        InputOutputTextPair(
            input_text="What is the base price of the Timeless package?",
            output_text="The base price of the Timeless Ceremony and Reception package is $6,299."
        ),
        InputOutputTextPair(
            input_text="What is the base price of the Forever package?",
            output_text="The base price of the Forever Ceremony and Reception package is $8,888."
        ),
        InputOutputTextPair(
            input_text="""How many photos are included in the Traditional Package?""",
            output_text="""The Traditional Package, priced at $495, comes with a dedicated photographer who will capture the ceremony. The number of photos varies, but you can discuss specific requirements with your assigned wedding planner."""
        ),
        InputOutputTextPair(
            input_text="""Is transportation provided in the Romantic Package?""",
            output_text="""The Romantic Package, starting at $650, includes essential services like flowers and photos. For transportation details or add-ons, it's best to consult directly with the chapel's coordinators."""
        ),
        InputOutputTextPair(
            input_text="""Can we bring our own decorations?""",
            output_text="""While Chapel of the Flowers provides comprehensive decor options, couples can bring specific personal touches with prior coordination."""
        ),
        InputOutputTextPair(
            input_text="""Can I get a basic package with additional flowers?""",
            output_text="""Yes, while the Basic Package starts at $299 and includes essential services, you can easily upgrade and add more floral arrangements as per your preference."""
        ),
        InputOutputTextPair(
            input_text="""How many photos are included in the Traditional Package?""",
            output_text="""The Traditional Package, priced at $495, comes with a dedicated photographer who will capture the ceremony. The number of photos varies, but you can discuss specific requirements with your assigned wedding planner."""
        ),
        InputOutputTextPair(
            input_text="""Is transportation provided in the Romantic Package?""",
            output_text="""The Romantic Package, starting at $650, includes essential services like flowers and photos. For transportation details or add-ons, it's best to consult directly with the chapel's coordinators."""
        ),
        InputOutputTextPair(
            input_text="""Can we have a rehearsal before the actual day?""",
            output_text="""Yes, Chapel of the Flowers facilitates rehearsals, ensuring everything goes smoothly on the big day."""
        ),
        InputOutputTextPair(
            input_text="""Do you have a backup generator?""",
            output_text="""Chapel of the Flowers is fully equipped with backup generators, ensuring uninterrupted celebrations regardless of external factors."""
        ),
        InputOutputTextPair(
            input_text="""Is there a waiting period after getting the license?""",
            output_text="""No, once you have your marriage license, Chapel of the Flowers can conduct the ceremony without any waiting period."""
        ),
        InputOutputTextPair(
            input_text="""Do you handle legal documentation post-wedding?""",
            output_text="""Chapel of the Flowers assists with essential documentation, guiding couples through the process for a seamless experience."""
        ),
        InputOutputTextPair(
            input_text="""Can we choose our own officiant from your team?""",
            output_text="""Yes, Chapel of the Flowers has a diverse team of officiants, and couples can express their preference to ensure a personal touch."""
        ),
        InputOutputTextPair(
            input_text="""Do you offer military discounts?""",
            output_text="""Chapel of the Flowers honors the service of military personnel and often provides special packages or discounts as a token of appreciation."""
        ),
        InputOutputTextPair(
            input_text="""How long is a typical ceremony?""",
            output_text="""A typical ceremony at Chapel of the Flowers lasts between 15 to 30 minutes, though it can be tailored to the couple's preferences."""
        ),
        InputOutputTextPair(
            input_text="""Can we have a reception at the venue?""",
            output_text="""Yes, Chapel of the Flowers can host receptions, offering customizable packages that cater to your desires and guest count."""
        ),
        InputOutputTextPair(
            input_text="""What's the cancellation policy?""",
            output_text="""Chapel of the Flowers has a comprehensive cancellation policy, which varies depending on the notice provided. It's advisable to consult directly for specifics."""
        ),
        InputOutputTextPair(
            input_text="""Are there photography restrictions?""",
            output_text="""While Chapel of the Flowers provides professional photography, certain areas or moments may have restrictions to ensure the sanctity and flow of the ceremony."""
        ),
        InputOutputTextPair(
            input_text="""Can we incorporate cultural rituals?""",
            output_text="""Absolutely! Chapel of the Flowers is experienced in various cultural rituals and customs, ensuring a rich and authentic celebration."""
        ),
        InputOutputTextPair(
            input_text="""Do I get a personal wedding planner with the $299 package?""",
            output_text="""Yes, the package includes a personal wedding planner to help plan and book all the amenities for your wedding."""
        ),
        InputOutputTextPair(
            input_text="""Can we include a unity candle ceremony in the Basic Package?""",
            output_text="""While the Basic Package offers foundational services, ceremonies like the unity candle can be integrated upon request, ensuring your wedding is tailored to your vision."""
        ),
        InputOutputTextPair(
            input_text="""Is our desired wedding date available? Do you have a backup plan in case of emergencies or unforeseen circumstances?""",
            output_text="""You can look at our availability calendar here: https://www.littlechapel.com/reservations/get-started."""
        ),
        InputOutputTextPair(
            input_text="""Can we review your standard contract? What is your cancellation policy and deposit refund conditions?""",
            output_text="""You can look at our availability calendar here: Yes, you can review our standard contract here: https://www.littlechapel.com/wp-content/uploads/2022/03/Little-Chapel-Contract.pdf."""
        ),
                InputOutputTextPair(
          input_text="""What time does the limousine pick up the wedding couple?""",
          output_text="""The limousine driver will call you in your hotel room approximately 45 to 60 minutes before your ceremony time to coordinate pick up details."""
          ),
        InputOutputTextPair(
          input_text="""How many guests are allowed at chapel wedding ceremonies?""",
          output_text="""The guest maximum varies depending on the capacity of the chapel you select."""
          ),
        InputOutputTextPair(
          input_text="""What happens when the couple arrives at the chapel on their wedding day?""",
          output_text="""When you arrive, you'll fill out paperwork, get your flowers, and meet with your minister and photographer to go over ceremony details."""
          ),
        InputOutputTextPair(
          input_text="""Are guests allowed to take photos during the ceremony?""",
          output_text="""We kindly ask that guests refrain from taking photos during the ceremony to allow you to be present in the moment."""
          ),
        InputOutputTextPair(
          input_text="""How long do couples get access to their ceremony photos online?""",
          output_text="""We keep your digital images for about 6 months, during which you can order reprints or prints."""
          ),
        InputOutputTextPair(
          input_text="""What happens after the ceremony concludes?""",
          output_text="""After the ceremony you'll meet with your minister to discuss marriage certificate details, then take photos if you reserved a photo session."""
          ),
        InputOutputTextPair(
          input_text="""Can we bring our own photographer?""",
          output_text="""Outside photography incurs a $500 fee. We'll provide guidance to your photographer on our policies."""
          ),
        InputOutputTextPair(
          input_text="""How do I get a copy of my marriage certificate?""",
          output_text="""Your minister files the documents with the county. You can then order certified copies for legal purposes."""
          ),
        InputOutputTextPair(
          input_text="""How do we obtain our marriage license?""",
          output_text="""You must get your Nevada marriage license at the County Clerk's office prior to your ceremony."""
          ),
        InputOutputTextPair(
          input_text="""What are your payment and deposit policies?""",
          output_text="""We require a $100 to $1000 deposit at booking, with the remaining balance due 30 days before your wedding."""
          ),
        InputOutputTextPair(
          input_text="""How can I arrange transportation for my guests?""",
          output_text="""We're happy to help you book limos, shuttles, or charter buses to transport your guests."""
          ),
        InputOutputTextPair(
          input_text="""What happens if we need to reschedule our ceremony?""",
          output_text="""One date change is free if notified 14+ days in advance. Additional terms apply."""
          ),
        InputOutputTextPair(
          input_text="""Do you allow outside alcohol to be served?""",
          output_text="""Unfortunately we cannot allow outside alcohol to be served at events."""
          ),
        InputOutputTextPair(
          input_text="""What type of reception chairs do you offer?""",
          output_text="""Our standard reception chairs are white chiavari chairs with cushioned seat pads."""
          ),
        InputOutputTextPair(
          input_text="""Do your packages include a wedding cake?""",
          output_text="""No, cake is not included but can be added. We have preferred bakers to create custom cakes."""
          ),
        InputOutputTextPair(
          input_text="""How long do ceremonies last?""",
          output_text="""The ceremony itself is typically 8-12 minutes long. Total chapel time is 30 minutes."""
          ),
        InputOutputTextPair(
          input_text="""What photography services are included?""",
          output_text="""Most packages include 30-90 minutes of photography with online image access."""
          ),
        InputOutputTextPair(
          input_text="""What floral options are available for ceremonies?""",
          output_text="""We offer aisle florals, altar arrangements, pew markers, petals, and more."""
          ),
        InputOutputTextPair(
          input_text="""Do you accommodate food allergies?""",
          output_text="""Yes, our caterers will work to accommodate any food allergies or restrictions."""
          ),
        InputOutputTextPair(
          input_text="""Are pets allowed at ceremonies?""",
          output_text="""Well-behaved pets are allowed at outdoor ceremonies with advance notice."""
          ),
        InputOutputTextPair(
          input_text="""Can you provide a wedding officiant?""",
          output_text="""Yes, we have licensed officiants on staff to perform your ceremony."""
          ),
        InputOutputTextPair(
          input_text="""Is there a comfortable waiting area for guests?""",
          output_text="""Yes, we have a luxurious lobby and outdoor space for guests to relax before ceremonies."""
          ),
        InputOutputTextPair(
          input_text="""Can I change my bridal bouquet style?""",
          output_text="""Yes, we can customize your bridal bouquet with advance notice."""
          ),
        InputOutputTextPair(
          input_text="""How can guests get back to hotels after the wedding?""",
          output_text="""Guests can take taxis waiting at the Stratosphere or rideshare services like Uber."""
          ),
        InputOutputTextPair(
          input_text="""Can we extend our photo session time?""",
          output_text="""Certainly, additional photography time can be purchased to capture more memories."""
          ),
        InputOutputTextPair(
          input_text="""Is parking available at the chapel?""",
          output_text="""Yes, we have limited parking available on-site for wedding couples and guests."""
          ),
        InputOutputTextPair(
          input_text="""What happens if my guests are late for the ceremony?""",
          output_text="""We'll do our best to delay the start if guests are running late, within reason."""
          ),
        InputOutputTextPair(
          input_text="""Can you recommend hotels for out-of-town guests?""",
          output_text="""We recommend hotels on the Strip or downtown for convenience to the chapel."""
          ),
        InputOutputTextPair(
          input_text="""Is the chapel wheelchair accessible?""",
          output_text="""Yes, our chapel and grounds are ADA compliant and wheelchair friendly."""
          ),
        InputOutputTextPair(
          input_text="""Do you allow outside catering?""",
          output_text="""We only allow approved caterers to maintain service standards."""
          ),
        InputOutputTextPair(
          input_text="""How do I arrange hair and makeup services?""",
          output_text="""Contact your planner to schedule hair/makeup with our preferred stylists."""
          ),
        InputOutputTextPair(
          input_text="""Can we have a rehearsal beforehand?""",
          output_text="""Unfortunately we don't offer rehearsals due to our wedding volume."""
          ),
        InputOutputTextPair(
          input_text="""Do you accommodate vegetarian or vegan meals?""",
          output_text="""Yes, our caterers can provide delicious vegetarian and vegan options."""
          ),
        InputOutputTextPair(
          input_text="""How are payments and deposits made?""",
          output_text="""We accept major credit cards and money orders for payments and deposits."""
          ),
        InputOutputTextPair(
          input_text="""How can I review my ceremony photos?""",
          output_text="""Your coordinator will schedule an online album viewing after your wedding."""
          ),
        InputOutputTextPair(
          input_text="""Do you allow flower petals to be scattered at ceremonies?""",
          output_text="""Yes, we allow biodegradable flower petals to be sprinkled."""
          ),
        InputOutputTextPair(
          input_text="""Is there a place to store wedding gifts and belongings?""",
          output_text="""We have a secure room to store gifts, cards, and other items."""
          ),
        InputOutputTextPair(
          input_text="""What happens if it rains on our wedding day?""",
          output_text="""Your planner will move the ceremony indoors in case of inclement weather."""
          ),
        InputOutputTextPair(
          input_text="""Can we extend our chapel time?""",
          output_text="""Certainly, additional chapel time can be purchased if needed."""
          ),
    ]
)

#MAKE SURE TO CHANGE PARAMS BEFORE EXECUTION!!!!!!
#MAKE SURE TO CHANGE PARAMS BEFORE EXECUTION!!!!!!
#MAKE SURE TO CHANGE PARAMS BEFORE EXECUTION!!!!!!

def chapel_chatbot(message, history):
    response = chat.send_message(message)
    return response.text

chapflow = gr.ChatInterface(
    fn=chapel_chatbot,
    title='Chapel of the Flowers',
    description='Experience VoodooVations -advanced AI-powered Natural Language Processing. Seamlessly integrate with your existing website and witness the clear advantages firsthand.',
    examples = ['What packages do offer?', 'Do you offer limo services?', 'What is included in the package of your choice?'],

  )

chapflow.launch(share=True)


