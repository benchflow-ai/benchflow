import Autoplay from 'embla-carousel-auto-scroll'
import useEmblaCarousel from 'embla-carousel-react'

export const QUESTIONS = [
  {
    title: 'Oswald\'s Lone Gunman Theory',
    content: 'Did Lee Harvey Oswald act alone in the assassination of President Kennedy, as concluded by the Warren Commission?',
  },
  {
    title: 'Conspiracy Evidence in JFK Files',
    content: 'Do the JFK files contain evidence of a conspiracy involving other individuals or groups in President Kennedy\'s assassination?',
  },
  {
    title: 'Oswald\'s Motivations',
    content: 'What were Lee Harvey Oswald\'s motivations for assassinating President Kennedy, and do the files shed new light on this?',
  },
  {
    title: 'Jack Ruby\'s Criminal Connections',
    content: 'What do the JFK files reveal about Jack Ruby\'s connections to organized crime and his motivations for killing Lee Harvey Oswald?',
  },
  {
    title: 'Foreign Powers Involvement',
    content: 'Do the JFK files provide any evidence of involvement by the Soviet Union or Cuba in President Kennedy\'s assassination?',
  },
  {
    title: 'CIA\'s Prior Knowledge',
    content: 'What did the CIA know about Lee Harvey Oswald prior to the assassination, and do the files suggest any level of involvement by the agency?',
  },
  {
    title: 'File Redactions Mystery',
    content: 'Why were certain JFK files withheld from the public for so long, and what might those redactions conceal?',
  },
  {
    title: 'Public Opinion Impact',
    content: 'How have the released JFK files impacted public opinion and the various conspiracy theories surrounding the assassination?',
  },
  {
    title: 'Historical Context',
    content: 'What significant historical insights into the political and social climate of the 1960s can be gained from examining the JFK files?',
  },
  {
    title: 'Warren Commission Validity',
    content: 'Do the JFK files offer any new evidence that strengthens or weakens the Warren Commission\'s conclusion that Lee Harvey Oswald was the lone assassin?',
  },
]

const AUTOPLAY_OPTIONS = {
  speed: 0.5,
  stopOnMouseEnter: false,
  stopOnInteraction: false,
} satisfies Parameters<typeof Autoplay>[0]

export function QuestionCarousel({ onQuestionClick }: {
  onQuestionClick: (question: string) => void
}) {
  const [rowRef, api] = useEmblaCarousel({
    loop: true,
    dragFree: true,
    direction: 'ltr',
  }, [Autoplay(AUTOPLAY_OPTIONS)])
  const autoplay = api?.plugins().autoScroll

  return (
    <div className="w-full space-y-8">
      <div ref={rowRef} className="overflow-hidden">
        <div className="flex">
          {QUESTIONS.map(question => (
            <div
              className="mr-2"
              key={question.title}
              onClick={() => onQuestionClick(question.content)}
              onMouseEnter={() => autoplay?.stop()}
              onMouseLeave={() => autoplay?.play()}
            >
              <button type="button" className="flex items-center px-4 py-2 border border-zinc-200 hover:border-slate-700 rounded-full bg-white/5 hover:bg-white/10 transition-colors">
                <span className="text-sm font-medium whitespace-nowrap">{question.title}</span>
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
