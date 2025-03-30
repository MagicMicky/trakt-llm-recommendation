Missing:

* Profile analysis improvements
  => A single, flat profile risks oversimplifying a complex, dynamic person.
  🧠 The Problem with a "One-Size" Profile
    A single profile might:
    Over-prioritize dominant patterns (e.g., Sci-fi, drama)
    Ignore smaller, meaningful interests (like that occasional comedy binge)
    Fail to capture contextual preferences (you don’t want the same shows after work vs. on a rainy weekend
    Lead to "filter bubble" fatigue — always recommending the same kind of show

  => Taste Clusters / Personas
    Divide the watch history into clusters (e.g., via genre or tone), like:
    “Cerebral sci-fi fan”
    “Dark, gritty crime thrillers”
    “Light-hearted comfort comedies”
    “High-production epics”

    You’re essentially saying: this user contains multitudes. The LLM can then:
    Match recommendations to each cluster
    Or recommend shows that bridge between them (e.g., a dark comedy thriller)



* Recommendations improvements
  => Recommendation should also follow these clusters.
  => LLM prompt should support those clusters.

* History fetching:
  => Not all shows are there?


* Feedback loop - is the show already viewed? Or ensure it's being excluded.

* Candidate fetching:
  => Only trending? 

* tech:
  * Clean up same code being used for openai etc.
  * Clean up logging so that we don't log to the console the openai prompt, only to file




* I think there's some issues in how some criterias are calculated.
  => Including binge


* It's taking all announced episode as total for show ie: doctor who says 16 but there's only 8