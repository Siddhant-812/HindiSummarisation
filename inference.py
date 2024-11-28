import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from model import ReRanker
import warnings

warnings.filterwarnings("ignore")

def generate_candidate_summaries(input_texts, num_candidates=4, model_name="csebuetnlp/mT5_multilingual_XLSum"):
    print("Generating candidate summaries...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

    candidate_summaries = []
    for text in tqdm(input_texts, desc="Generating Summaries"):
        encoded_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = model.generate(
            input_ids=encoded_input["input_ids"],
            attention_mask=encoded_input["attention_mask"],
            num_return_sequences=num_candidates,
            num_beams=num_candidates,
            num_beam_groups=4, 
            no_repeat_ngram_size=2,
            max_length=80,
            diversity_penalty=1.0,  
            early_stopping=True,
        )
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        candidate_summaries.append(decoded_outputs)

    return candidate_summaries


def rank_summaries(input_texts, candidate_summaries, model_path, tokenizer_name="ai4bharat/indic-bert", device="cuda"):
    print("Ranking candidate summaries...")
    # Load tokenizer and trained ReRanker model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = ReRanker(model_path, pad_token_id=tokenizer.pad_token_id).to(device)
    model.encoder.load_adapter(model_path,"adapter_model")  # Load trained LoRA adapter
    model.eval()

    ranked_results = []
    for i, (text, candidates) in tqdm(enumerate(zip(input_texts, candidate_summaries)), desc="Ranking Summaries", total=len(input_texts)):
        # Encode input text and candidates
        encoded_input = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
        encoded_candidates = tokenizer(candidates, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)

        # Perform ranking
        with torch.no_grad():
            outputs = model(
                text_id=encoded_input["input_ids"],
                candidate_id=encoded_candidates["input_ids"],
                require_gold=False,  # No gold summaries during inference
            )
            scores = outputs["score"].squeeze(0)  # Shape: (num_candidates,)
            ranked_indices = torch.argsort(scores, descending=True).tolist()  # Sort candidates by scores

        # Append the best candidate
        # ranked_summaries.append([candidates[i] for i in ranked_indices])
        ranked_candidates = [{"summary": candidates[i], "score": scores[i].item()} for i in ranked_indices]
        ranked_results.append(ranked_candidates)


    return ranked_results


if __name__ == "__main__":
    # Hardcoded input texts
    input_texts = [
        "नासिक की एक अदालत ने महाराष्ट्र पुलिस के आतंकवाद निरोधक दस्ते (एटीएस) के अनुरोध पर ये आदेश दिया है. सरकारी वकील अजय मिश्रा ने यह जानकारी दी है. कुछ दिन पहले एटीएस ने मध्य प्रदेश के इंदौर शहर से साध्वी प्रज्ञा सिंह ठाकुर को गिरफ़्तार किया था. नासिक की अदालत ने प्रज्ञा सिंह और दो अन्य अभियुक्तों को तीन नवंबर तक के लिए पुलिस हिरासत में रखने का भी फ़ैसला दिया. आवेदन एटीएस ने अदालत में इस टेस्ट के लिए आवेदन किया था और कहा था कि जाँच के लिए ये टेस्ट ज़रूरी हैं. हालाँकि सरकारी वकील अजय मिश्रा ने यह जानकारी नहीं दी कि साध्वी प्रज्ञा सिंह ठाकुर को कब इस टेस्ट से गुज़रना होगा. पत्रकारों के एक सवाल के जवाब में उन्होंने बताया कि उन्हें इस बात की कोई जानकारी नहीं है कि इसी मामले में दो पूर्व सैनिक अधिकारियों को भी अदालत में पेश किया जाना है. इन दो पूर्व सैनिक अधिकारियों से मालेगाँव और सूरत के मोडासा में हुए धमाकों के सिलसिले में पूछताछ की जा रही है. बीबीसी संवाददाता रेहाना बस्तीवाला ने पुलिस सूत्रों के हवाले से इसकी पुष्टि की है और बताया है कि इनमें एक रिटायर्ड मेजर जनरल भी है. मालेगाँव में 29 सितंबर को धमाका हुआ था. धमाका और उसके बाद हुई पुलिस गोलीबारी में पाँच लोगों की मौत हो गई थी.",
        "अगले साल फरवरी में होने वाली चैंपियंस ट्रॉफी पाकिस्तान में होगी या नहीं, इसका फैसला 29 नवंबर को होगा। ESPN की रिपोर्ट के मुताबिक, ICC ने दुबई में बोर्ड मीटिंग बुलाई है। पाकिस्तान को चैंपियंस ट्रॉफी की मेजबानी मिलने के बाद भारत ने सुरक्षा कारणों का हवाला देकर वहां जाने से मना कर दिया था। तब यह माना जा रहा था कि एशिया कप की तरह चैंपियंस ट्रॉफी भी हाइब्रिड मॉडल पर होगी। पाकिस्तान क्रिकेट बोर्ड (PCB) ने पहले भारत के सभी मैच लाहौर में कराने और मैच के बाद खिलाड़ियों को भारत भेजने का प्रस्ताव रखा था। भारत ने इसे नहीं माना तो (PCB) ने हाइब्रिड मॉडल के लिए भी मना कर दिया। ICC मीटिंग में हाइब्रिड मॉडल का प्रस्ताव रख सकता है, अगर PCB ने इसे नहीं माना तो उससे मेजबानी छिन सकती है। भारत पर पाकिस्तान जाकर खेलने का दबाव बनाने के चांस कम ही हैं। 2008 में मुंबई हमले के बाद से ही भारतीय टीम पाकिस्तान नहीं गई है।",
        "16 महीने के कार्यकाल पूरे होने के बावजूद राष्ट्रपति मैक्रों आर्थिक वृद्धि और रोज़गार के वादे को पूरा नहीं कर सके हैं. मैक्रों अपने ही मुल्क में लोकप्रियता खो रहे हैं. कौन सा डर मैक्रों को सता रहा है? पोल एजेंसियों का कहना है कि इमैनुएल मैक्रों की लोकप्रियता सबसे निचले स्तर पर पहुंच चुकी है. ओपिनियनवे के मुताबिक मौजूदा वक़्त में फ़्रांस के केवल 28 फ़ीसदी मतदाता ही उनके कामकाज से संतुष्ट हैं. जुलाई में यह 35 फ़ीसदी था. इन आंकड़ों के मुताबिक अपने कार्यकाल के इस समान अवधि में मैक्रों की लोकप्रियता पूर्व राष्ट्रपति फ़्रास्वां ओलांद और निकोलस सरकोज़ी से भी कम है. इन आंकड़ों पर साइंसेज़ पो यूनिवर्सिटी के प्रोफ़ेसर क्रिसटोफी दे फॉक्ड ने कहते हैं, ''इस बात का ख़्याल रखना होगा कि मैक्रों की चुनावी जीत बहुत बड़ी नहीं थी. ज़्यादातर लोगों ने मैक्रों को इसलिए चुना क्योंकि उनकी छवि अति दक्षिणपंथी नहीं थी. इसलिए उनके ठोस मतदाताओं की संख्या कम है.'' ट्रंप और मैक्रों ने नए ईरान परमाणु समझौते के दिए संकेत वो आगे कहते हैं, ''एक बात का हमें ख़्याल रखना होगा कि फ़्रांस एक आसान देश नहीं है. यहां राजनैतिक असंतोष का इतिहास रहा है. ऐसे में यहां नेताओं की नकरात्मक रेटिंग लाजिमी है.'' आख़िर मैक्रों से ग़लती कहां हुई? प्रोफ़ेसर दे फ़ॉक्ड इसके तीन बिंदु बताते हैं. मैक्रों के बड़े-बड़े वादे फ़्रांस के मतदाताओं ने आर्थिक वृद्धि में सकारात्मक पहल की उम्मीद की थी, लेकिन ये हुआ नहीं. प्रोफ़ेसर दे फ़ॉक्ड कहते हैं, ''जब आपसे बहुत ज़्यादा उम्मीद हो और आप धरातल पर कुछ ना कर सकें तो जनता की ऐसी निराशा का सामना आपको करना पड़ता है.'' मैक्रों की साख़ को एलेक्ज़ेंड्रे बैनेला मामले में सबसे ज़्यादा नुकसान होगा. इस साल जुलाई में मैक्रों के सुरक्षाकर्मी 26 वर्षीय एलेक्ज़ेंड्रे बैनेला का एक वीडियो सामने आया था, जिसमें वो एक प्रदर्शनकारी को मारते नज़र आ रहे थे. फ्रांस के राष्ट्रपति ने अमरीका में किया 'राष्ट्रवाद' पर हमला मीडिया की तमाम आलोचनाओं के बाद एलेक्ज़ेंड्रे को फ़्रांस सरकार ने पद से हटाया. इसके बाद से ही सरकार पर सवाल उठने लगे कि आख़िर सब कुछ जानते हुए भी मैक्रों ने क्यों एलेक्ज़ेंड्रे बैनेला को बचाने की कोशिश की? प्रोफ़ेसर दे फॉक्ड बताते हैं, ''अगर ये घटना ब्रिटेन में हुई होती तो इस मामले में गृह सचिव का इस्तीफ़ा हो चुका होता. लेकिन फ़्रांस में ऐसा कुछ भी नहीं हुआ. इस घटना के बाद फ़्रांस के लोगों को ये लगा कि कुछ कुलीन लोगों के लिए सरकार के अलग नियम हैं और बाक़ी देश के लिए अलग नियम लागू किया जा रहा है.'' मैक्रों पर क्या कहते हैं लोग? मतदाताओं के किसी भी समूह से अगर मैक्रों के लिए एक विशेषण का इस्तेमाल करने को कहा जाए तो वे 'घमंडी' शब्द का इस्तेमाल करते हैं. कई बार कैमरे पर मैक्रों अपने देश के नागरिकों के लिए असंवेदनशील और अपमानजनक टिप्पणी कर चुके हैं. इसका सबसे ताज़ातरीन उदाहरण है एलिसी में हालिया आयोजित हुआ 'ओपेन डे'. वहां उन्होंने बाग़ान के मालियों के लिए कहा था, 'जो माली जो काम ना मिलने की शिकायत करते हैं उन्हें अन्य व्यवसाय की ओर रुख़ करना चाहिए. वे खाना परोसने का काम सीख सकते हैं.' हालांकि कुछ लोगों का मानना है कि ये एक कठोर सच है, जिसे राष्ट्रपति ने कहा. लेकिन फ़्रांस के ज़्यादातर लोगों का मानना है कि ये असंवेदनशील बयान है. इप्सोस पोल एजेंसी की विशेषज्ञ क्लो मॉरिन कहती हैं, ''मैक्रों अक़्सर इस तरह के बयान देते रहते हैं जो लोगों को उनके धमंडी होने का संदेश देता है. ये बयान उनकी नकारात्मक छवि बना रहे हैं.'' बदलनी होगी आर्थिक स्थिति इमैनुएल मैक्रों के कार्यकाल के अभी साढ़े तीन साल बचे हुए हैं. ये पर्याप्त समय है जिसमें वो देश की आर्थिक स्थिति को पलट सकते हैं. ये जानना बेहद अहम है कि फ़्रांस की राजनीति में राष्ट्रपति बेहद शक्तिशाली पद है. वह जो चाहे वो करने का अधिकार रखता है. देश में मैक्रों की नीतियों से लोग नाराज़ हैं, लेकिन दूसरी ओर अंतरराष्ट्रीय स्तर पर मैक्रों ने जिस तरह फ़्रांस की तस्वीर बदली है इसकी लोग तारीफ़ करते हैं. फ्रांस के कमज़ोर विपक्ष का भी फ़ायदा मैक्रों उठा सकते हैं. मैक्रों के बारे में कहा जाता है कि वो अपनी आलोचनाओं पर प्रतिक्रिया देते हुए फ़ैसले लेते हैं. हाल ही में लोगों का समर्थन पाने के लिए नए टैक्स मानकों का ऐलान ऐसा ही एक क़दम माना जाता है. मैक्रों की नज़र यूरोपीय संघ के चुनाव पर भी होगी. पिछले हफ्ते सामने आए एक पोल के मुताबिक मैक्रों और मैरिन इस टक्कर में काफ़ी क़रीब हैं. अगर ये चुनाव मैरिन जीत जाती हैं तो ये मैक्रों के लिए बड़ा झटका होगा. ये भी पढ़ें जब संयुक्त राष्ट्र में ट्रंप का भाषण सुनकर हंस पड़े लोग आखिर क्यों घट रही है फ्रांसीसी राष्ट्रपति मैक्रों की लोकप्रियता फ़्रांसीसी राष्ट्रपति इमैनुएल मैक्रों: एक ट्रंप विरोधी (बीबीसी हिन्दी के एंड्रॉएड ऐप के लिए आप यहां क्लिक कर सकते हैं. आप हमें फ़ेसबुक, ट्विटर, इंस्टाग्राम और यूट्यूब पर फ़ॉलो भी कर सकते हैं.)",
        '"या ख़ुदा! अब हम अमरीका आ जा सकते हैं! मैं पहनूँगी क्या!" ईरान और दुनिया के छह प्रमुख देशों के बीच हो रहा परमाणु समझौता एक गंभीर विषय है. लेकिन कुछ आम ईरानी इसके संभावित परिणामों को लेकर इस तरह की हँसी-ठिठोली कर रहे हैं. इस समझौते के बाद ईरान और अमरीका के बीच जिस तरह का संभावित गठजोड़ होगा उसे लेकर ईरान में मोबाइल संदेशों और सोशल मीडिया पर नए-नए चुटकुले बनाए जा रहे हैं. एक चुटकुले में कहा गया है, "जैसे ओबामा के नाम पर मैनहट्टन का नाम बदलकर मैश हसन (राष्ट्रपति हसन रूहानी के नाम पर) कर दिया गया, वैसे ही रूहानी को भी अरक (शहर) का नाम बराक रखने का आदेश देना चाहिए." समाप्त \'अमरीकी योग्य वर\' ईरानी सोशल मीडिया पर शेयर किया गया एक चुटकुला. ईरान और अमरीका के एक साथ आ जाने की उम्मीद से आम ईरानियों में उत्साह का माहौल है. उन्हें लग रहा है कि ये ईरान के लंबे राजनीतिक अलगाव के ख़त्म होने का लक्षण है. ईरान में अमरीकी नौजवानों के योग्य वर बनकर आने की संभावना पर भी चुटकियाँ ली जा रही हैं. ज़्यादातर चुटकुलों में अमरीका और ईरान के बीच समझौता हो जाने के बाद आम ईरानियों के रोज़मर्रा के जीवन पर पड़ने वाले प्रभावों को लेकर मज़ाक किया जा रहा है. मसलन एक व्यक्ति ने लिखा है, "मैंने प्राइड (स्थानीय कार ब्रांड) ख़रीदने के लिए 20 लाख तोमान (सात हज़ार डॉलर) बचाए थे. अब इस समझौते के बाद मैं सोच रहा हूँ कि पोर्श ख़रीदूँ या मैसेराती?" अमरीका समर्थिक ईरान के शाह के 1979 की क्रांति के बाद सत्ता से बेदखल होने के बाद से दोनों देशों के रिश्ते खट्टे रहे हैं. नकाब पर \'मज़ाक\' एक परंपरागत रूढ़िवादी देश जहाँ आम तौर पर महिलाएँ इस्लामी नकाब पहनती हैं, वहाँ चमत्कारिक रूप से एक उदारवादी देश में बदल जाने की संभावना पर भी मज़ाक किया जा रहा है. एक यूज़र ने लिखा है, "प्यारे दोस्तों, ये समझौता परमाणु मुद्दे पर हुआ है! कृपया इसे दूसरों को भी बताएँ! लोग तो सड़कों पर शॉर्ट्स और टैंक टॉप्स पहनकर निकल रहे हैं!" (बीबीसी हिन्दी के एंड्रॉएड ऐप के लिए आप यहां क्लिक कर सकते हैं. आप हमें फ़ेसबुक और ट्विटर पर फ़ॉलो भी कर सकते हैं.)',
        'इसराइल की ख़ुफ़िया एजेंसी मोसाद को दुनिया भर में सर्वश्रेष्ठ एजेंसियों में गिना जाता है. इसराइल की एक संसदीय समिति ने जाँच के बाद कहा है कि मोसाद और सैनिक ख़ुफ़िया एजेंसियाँ यह बताने में नाकाम रहीं कि इराक़ के पास महाविनाश के हथियार थे या नहीं. समिति का कहना है और एजेंसियों की यह नाकामी उनके स्तर में भारी गिरावट का सबूत है. इस जाँच रिपोर्ट में कहा गया है कि इसराइल ने पिछले साल इराक़ पर हमले से पहले इराक़ की सैन्य क्षमता के बारे में अमरीका और ब्रिटेन को जानबूझकर गुमराह नहीं किया. समिति का कहना है कि ख़ुफ़िया एजेंसियों ने ग़लती से यह नतीजा निकाल लिया कि इराक़ के पास महाविनाश के हथियार हैं. समिति ने सरकार की इस बात के लिए भी आलोचना की है कि उसने इराक़ पर हमले के दौरान लोगों इसराइल के लोगों को अपने गैस मास्क इस्तेमाल करने का आदेश दिया जिस पर दो करोड़ डॉलर से भी ज़्यादा धन बर्बाद हुआ. लीबिया लीबिया के मामले में भी कहा गया है कि ख़ुफ़िया एजेंसियों को लीबिया के परमाणु कार्यक्रम के बारे में भी कुछ पता नहीं चला. उन्हें तभी पता चला जब ख़ुद लीबिया ने ही यह राज़ खोला. संसदीय समिति ने लीबिया के बारे में एजेंसियों की इस नाकामी को अस्वीकार्य बताया है. संसदीय समिति की इस रिपोर्ट में इसराइल की ख़ुफ़िया एजेंसियों में व्यापक फेरबदल की सिफ़ारिश की है. इस समिति में सभी दलों के सदस्य थे.',
        "फ़ाइल फ़ोटो (17 अगस्त 2013, मिस्र) इस गंभीर दावे के साथ 30 सेकेंड का एक वीडियो सोशल मीडिया पर शेयर किया जा रहा है. वीडियो में कुछ लोग चर्च के मुख्य द्वार के ऊपर चढ़े हुए दिखाई देते हैं और वीडियो का अंत होते-होते वो चर्च के धार्मिक चिह्न को तोड़कर नीचे गिरा देते हैं. वीडियो में लोगों के चिल्लाने की आवाज़ सुनी जा सकती है और इसके एक हिस्से में चर्च की इमारत से धुआँ उठता हुआ भी दिखाई देता है. फ़ेसबुक और ट्विटर पर अभी इस वीडियो को कम ही लोगों ने शेयर किया है, लेकिन व्हॉट्सऐप के ज़रिए बीबीसी के कई पाठकों ने हमें यह वीडियो भेजकर इसकी सत्यता जाननी चाही है. यूके के लंदन शहर में रहने वाली एक ट्विटर यूज़र @TheaDickinson ने भी इस वीडियो को पोस्ट करते हुए यही दावा किया है. उन्होंने यह सवाल भी उठाया है कि बीबीसी ने इस वीडियो को क्यों नहीं दिखाया? लेकिन 'पाकिस्तान के चर्च में आग लगाए जाने' के इस दावे को अपनी पड़ताल में हमने फ़र्ज़ी पाया है. वायरल वीडियो क़रीब 6 साल पुराना है. वीडियो पाकिस्तान का नहीं न्यूज़ीलैंड के क्राइस्टचर्च की दो मस्जिदों (अल नूर और लिनवुड मस्जिद) में 15 मार्च को ब्रेंटन टैरंट नाम के एक हमलावर ने गोलीबारी की थी. इस घटना में क़रीब 50 लोगों की मौत हो गई थी और 50 से ज़्यादा लोग घायल हुए थे. फ़ाइल फ़ोटो (20 अगस्त 2013, मिस्र) न्यूज़ीलैंड की प्रधानमंत्री जैसिंडा अर्डर्न मस्जिद में हुए इस हमले को 'आतंकवादी हमला' और देश के लिए 'काला दिन' बता चुकी हैं. लेकिन जिस 30 सेकेंड के वीडियो को क्राइस्टचर्च हमले के 'बदले का वीडियो' बताया जा रहा है वो साल 2013 का वीडियो है. रिवर्स इमेज सर्च से पता चलता है कि ये वीडियो पाकिस्तान का भी नहीं है, बल्कि मिस्र का है. यू-ट्यूब पर 29 अगस्त 2013 को पब्लिश किये गए 6:44 सेकेंड के एक वीडियो में वायरल वीडियो का 30 सेकेंड का हिस्सा दिखाई देता है. फ़ाइल फ़ोटो (21 अगस्त 2013, मिस्र) कॉप्टिक चर्चों पर हमला अगस्त 2013 में मिस्र के कम से कम 25 चर्चों में ईसाई-विरोधी गुटों ने हिंसा की थी. ये वायरल वीडियो उसी समय का है. साल 2013 में ही कॉप्टिक ऑर्थोडॉक्स चर्च को भी निशाना बनाया गया था जिसके बारे में मान्यता है कि ये पचासवीं ईस्वी के आसपास बना था और अलेक्जेंड्रिया में स्थापित ईसाई धर्म के सबसे पुराने चर्चों में से एक रहा है. मिस्र के पूर्व राष्ट्रपति मोहम्मद मोर्सी के तख़्ता पलट को ईसाई विरोधी हिंसा का मुख्य कारण माना जाता है. मिस्र के पूर्व राष्ट्रपति मोहम्मद मोर्सी की तस्वीर (फ़ाइल फ़ोटो) जुलाई 2013 में सेना के मिस्र पर क़ब्ज़ा कर लेने के बाद जब जनरल अब्दुल फ़तेह अल-सीसी ने टीवी पर राष्ट्रपति मोर्सी के अपदस्थ होने की घोषणा की थी, तब पोप टावाड्रोस द्वितीय उनके साथ खड़े नज़र आए थे. उसके बाद से ही ईसाई समुदाय के लोग कुछ इस्लामिक कट्टरपंथियों के निशाने पर रहे हैं. तख़्ता पलट के समय पोप ने कहा था कि जनरल सीसी ने मिस्र का जो रोडमैप (ख़ाका) दिखाया है, उसे मिस्र के उन सम्मानित लोगों द्वारा तैयार किया गया है जो मिस्र का हित चाहते हैं. फ़ाइल फ़ोटो (27 अगस्त 2013, मिस्र) पोप के इस बयान के बाद उन्हें कई दफ़ा मारने की धमकी दी गई थी. जबकि कई ईसाइयों की हत्या कर दी गई थी और उनके घरों को निशाना बनाया गया था. मिस्र के अधिकतर ईसाई कॉप्टिक हैं जो प्राचीन मिस्रवासियों के वंशज हैं. मिस्र की कुल जनसंख्या में लगभग दस प्रतिशत ईसाई हैं और सदियों से सुन्नी बहुल मुसलमानों के साथ शांति से रहते आए हैं. (इस लिंक पर क्लिक करके भी आप हमसे जुड़ सकते हैं) (बीबीसी हिन्दी के एंड्रॉएड ऐप के लिए आप यहां क्लिक कर सकते हैं. आप हमें फ़ेसबुक, ट्विटर, इंस्टाग्राम और यूट्यूब पर फ़ॉलो भी कर सकते हैं.)",

    ]

    # Configuration
    model_path = "./output"  # Path to the trained ReRanker model
    tokenizer_name = "ai4bharat/indic-bert"  # IndicBERT tokenizer
    num_candidates = 4  # Number of candidate summaries to generate
    output_file = "ranked_summaries.json"  # File to save ranked summaries

    # Step 1: Generate candidate summaries
    candidate_summaries = generate_candidate_summaries(input_texts, num_candidates=num_candidates)

    # Step 2: Rank candidate summaries
    ranked_summaries = rank_summaries(input_texts, candidate_summaries, model_path)

    # Save the ranked summaries to a JSON file
    import json
    with open(output_file, "w") as f:
        json.dump(ranked_summaries, f, ensure_ascii=False, indent=4)

    print(f"Ranked summaries saved to {output_file}")