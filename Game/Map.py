import asyncio
class Map:

    def __init__(self,agent,owned_villages=[],other_villages=[]):
        """
        We are supposing that it's a single player game so we are going to set the Agent here
        :param owned_villages:
        :param other_villages:
        """
        self.owned_villages = owned_villages
        self.other_villages = other_villages
        self.agent = agent

    async def choose_actions(self,village):
        while True:
            #First we are going to poll all actions
            all_upgrades,_ = village.get_available_upgrades()
            all_upgrades["idle"] = 1
            action = self.agent.choose_action(all_upgrades)
            print("Performing action",action)
            village.perform_action(action)
            await asyncio.sleep(1)


    async def run_game(self):
        async with asyncio.TaskGroup() as tg:
            for village in self.owned_villages:
                tg.create_task(village.run(0.01))
                tg.create_task(self.choose_actions(village))
        print("Both tasks have completed now.")

