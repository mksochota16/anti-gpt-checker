from config import init_all_polish_models
from analysis.nlp_transformations import preprocess_text
from analysis.attribute_retriving import perform_full_analysis

if __name__ == "__main__":
    init_all_polish_models()

    test_text = """
    W cichym miasteczku Eldridge jesień zaczynała malować ulice w odcieniach bursztynu i złota. Liście, 
    chrupiące pod stopami, szumiały o nadchodzącej zmianie pory roku, a dzieci goniły się po parkach, 
    ich śmiech odbijał się echem od starych murów z cegły. Wśród mieszkańców panowało poczucie oczekiwania, 
    delikatne podekscytowanie nadchodzącym Festiwalem Plonów, który obiecywał ucztę smaków, dźwięków 
    i tradycji, obchodzonych od pokoleń.

    Pan Thompson, miejscowy piekarz, budził się przed świtem, by zagniatać ciasto na chleb dyniowy, 
    jego aromat wkrótce rozchodził się po miasteczku, mieszając z subtelnym zapachem dymu z kominów. 
    Po drugiej stronie ulicy rodzina Millerów przygotowywała szarlotki, krojąc jabłka z uwagą i posypując 
    je cukrem, a para unosząca się z piekarników wplatała się w chłodne poranne powietrze. 
    W bibliotece pani Eleonora odkurzała stare tomy, w których zapisane były historie Eldridge, 
    każda strona opowiadała o dawnych czasach, gdy ulice były pełne handlarzy, a place tętniły życiem.

    W miarę jak słońce wstawało coraz wyżej, mieszkańcy zaczynali dekorować place i sklepy, 
    wieszając girlandy z suszonych kwiatów i dyni, przygotowując stoiska pełne świeżych warzyw, 
    owoców i wyrobów rzemieślniczych. Dzieci z dumą prezentowały swoje rzeźby z dyni, 
    a starzy mieszkańcy wspominali minione lata, opowiadając młodszym historie o dawnych festiwalach, 
    o zabawach, konkursach i o tym, jak zmieniało się miasto wraz z kolejnymi pokoleniami. 

    Każdy zaułek Eldridge pachniał inaczej: zapach pieczonych jabłek mieszał się z aromatem świeżego chleba, 
    a czasami z nutą wilgotnej ziemi i liści, które spadały z drzew. W powietrzu unosiła się atmosfera radości, 
    ale i pewnej nostalgii, bo mieszkańcy wiedzieli, że każda jesień przypomina o przemijaniu i o tym, że 
    czas nie stoi w miejscu. W kawiarni „Pod Dębem” pan Kowalski przygotowywał gorącą czekoladę, 
    rozlewając ją do filiżanek i posypując szczyptą cynamonu, który natychmiast wypełniał wnętrze przytulnym zapachem.

    Na obrzeżach miasteczka młodzież zbierała dynie z pól, śmiejąc się i przewracając w stosach liści, 
    a starsi mieszkańcy pomagali w ustawianiu stoisk i przygotowywaniu dekoracji. 
    Wszyscy wiedzieli, że Festiwal Plonów to nie tylko okazja do zabawy, ale też moment refleksji nad 
    tym, co udało się osiągnąć przez cały rok. Dzień przechodził w wieczór, a światła lamp rozświetlały ulice, 
    tworząc magiczną atmosferę, która sprawiała, że każdy czuł się częścią wspólnoty, częścią historii Eldridge.

    Im bliżej było nocy, tym więcej osób przybywało na główny plac. Zaczęły się koncerty lokalnych zespołów, 
    dzieci uczestniczyły w grach i konkursach, a stoiska oferowały smakołyki, które kusiły zapachem i wyglądem. 
    Rozmowy, śmiech, muzyka i dźwięki tradycyjnych instrumentów tworzyły harmonijną mieszankę, której 
    nie dało się doświadczyć nigdzie indziej. Każdy krok po brukowanych uliczkach przypominał o 
    bogactwie wspomnień, o życiu, które toczyło się przez lata i o ludziach, którzy nadawali mu sens.

    W nocy niebo nad Eldridge przybrało kolor głębokiego granatu, a gwiazdy lśniły jasno, jakby chciały 
    wskazać drogę wszystkim podróżnikom. Dźwięki ostatnich koncertów i rozmów powoli cichły, 
    pozostawiając miejsce na ciszę, która była równie pełna znaczenia jak dzień. 
    Mieszkańcy wracali do domów, niosąc ze sobą wspomnienia, uśmiechy i poczucie, 
    że uczestnictwo w Festiwalu Plonów to coś więcej niż zwykła zabawa – to rytuał, 
    który łączy pokolenia i pozwala na chwilę refleksji nad tym, co naprawdę ważne w życiu.
    """

    text_to_analyse = preprocess_text(test_text)
    analysis_result = perform_full_analysis(text_to_analyse, 'pl')

    print(analysis_result.dict())
