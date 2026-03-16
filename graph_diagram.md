---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	initialize_game(initialize_game)
	deal_cards(deal_cards)
	chat_start(chat_start)
	ai_chat(ai_chat)
	human_chat(human_chat)
	route_player(route_player)
	ai_decision(ai_decision)
	human_input(human_input)
	execute_action(execute_action)
	advance_street(advance_street)
	showdown(showdown)
	check_game_over(check_game_over)
	__end__([<p>__end__</p>]):::last

	__start__ --> initialize_game
	initialize_game --> deal_cards
	deal_cards --> chat_start

	chat_start -. ai_chat .-> ai_chat
	chat_start -. human_chat .-> human_chat
	chat_start -. chat_done .-> route_player

	ai_chat -.-> ai_chat
	ai_chat -.-> human_chat
	ai_chat -. chat_done .-> route_player

	human_chat -.-> ai_chat
	human_chat -.-> human_chat
	human_chat -. chat_done .-> route_player

	route_player -. ai_decision .-> ai_decision
	route_player -. human_input .-> human_input

	ai_decision --> execute_action
	human_input --> execute_action

	execute_action -. continue_betting .-> route_player
	execute_action -. street_complete .-> advance_street
	execute_action -. only_one_active .-> showdown

	advance_street -. deal_cards .-> deal_cards
	advance_street -. showdown .-> showdown

	showdown --> check_game_over

	check_game_over -. new_round .-> initialize_game
	check_game_over -. end .-> __end__

	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
	classDef default fill:#f2f0ff,line-height:1.2
