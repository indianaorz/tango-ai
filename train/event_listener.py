# event_listener.py

import asyncio
import json

async def event_listener(instance):
    server = await asyncio.start_server(
        lambda r, w: handle_game_event(r, w, instance),
        '127.0.0.1',
        instance['event_port']
    )
    async with server:
        await server.serve_forever()

async def handle_game_event(reader, writer, instance):
    while True:
        data = await reader.readline()
        if not data:
            break
        event = json.loads(data.decode())
        process_event(event, instance)
    writer.close()
    await writer.wait_closed()

def process_event(event, instance):
    env = instance['env']
    if event['event_type'] == 'damage_dealt':
        env.damage_dealt += event['value']
    elif event['event_type'] == 'damage_received':
        env.damage_received += event['value']
    elif event['event_type'] == 'game_over':
        env.done = True
