export interface Model {
  id: string
  name: string
}

export interface BattleResponse {
  response1: string
  response2: string
  battle_id: number
}

export interface SelectedModels {
  model1: Model | null
  model2: Model | null
}

export type VoteResult = 'model1' | 'model2' | 'draw' | 'invalid'
