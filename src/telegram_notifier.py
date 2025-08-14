"""
Telegram notification system for audio monitoring alerts.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime
from telegram import Bot
from telegram.error import TelegramError


class TelegramNotifier:
    """
    Telegram bot for sending audio detection notifications.
    """
    
    def __init__(self, bot_token: str, chat_id: str, cooldown_seconds: float = 5.0):
        """
        Initialize the Telegram notifier.
        
        Args:
            bot_token: Telegram bot token
            chat_id: Telegram chat ID to send messages to
            cooldown_seconds: Minimum time between notifications for same sound type
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.cooldown_seconds = cooldown_seconds
        
        self.bot: Optional[Bot] = None
        self.last_notification_times: Dict[str, float] = {}
        
        self.logger = logging.getLogger(__name__)
        
        self._initialize_bot()
    
    def _initialize_bot(self) -> None:
        """Initialize the Telegram bot."""
        try:
            # Initialize bot with connection pool settings
            from telegram.request import HTTPXRequest
            request = HTTPXRequest(
                connection_pool_size=20,
                pool_timeout=30.0,
                read_timeout=30.0,
                write_timeout=30.0
            )
            self.bot = Bot(token=self.bot_token, request=request)
            self.logger.info("Telegram bot initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Telegram bot: {e}")
            raise
    
    async def _send_message_async(self, message: str) -> bool:
        """
        Send a message asynchronously.
        
        Args:
            message: Message text to send
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
            return True
            
        except TelegramError as e:
            self.logger.error(f"Telegram error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def send_notification(
        self,
        sound_type: str,
        confidence: float,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Send a notification about detected sound.

        Args:
            sound_type: Type of sound detected
            confidence: Confidence score (0.0 to 1.0)
            timestamp: When the sound was detected (defaults to now)

        Returns:
            True if notification was sent, False otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Check cooldown
        current_time = time.time()
        last_time = self.last_notification_times.get(sound_type, 0)

        if current_time - last_time < self.cooldown_seconds:
            self.logger.debug(
                f"Skipping notification for {sound_type} due to cooldown "
                f"({current_time - last_time:.1f}s < {self.cooldown_seconds}s)"
            )
            return False

        # Create notification message
        message = self._create_notification_message(sound_type, confidence, timestamp)

        # Send message with better error handling
        try:
            # Try to get existing event loop, create new one if needed
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("Event loop is closed")
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run the async function
            success = loop.run_until_complete(self._send_message_async(message))

            if success:
                self.last_notification_times[sound_type] = current_time
                self.logger.info(f"Sent notification for {sound_type} (confidence: {confidence:.2f})")

            return success

        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
            return False
    
    def _create_notification_message(
        self,
        sound_type: str,
        confidence: float,
        timestamp: datetime
    ) -> str:
        """
        Create a formatted notification message.
        
        Args:
            sound_type: Type of sound detected
            confidence: Confidence score
            timestamp: Detection timestamp
            
        Returns:
            Formatted message string
        """
        # Get appropriate emoji for sound type
        emoji = self._get_sound_emoji(sound_type)
        
        # Format confidence as percentage
        confidence_pct = confidence * 100
        
        # Format timestamp
        time_str = timestamp.strftime("%H:%M:%S")
        date_str = timestamp.strftime("%Y-%m-%d")
        
        message = (
            f"🚨 <b>Audio Alert</b> 🚨\n\n"
            f"{emoji} <b>Sound Detected:</b> {sound_type}\n"
            f"📊 <b>Confidence:</b> {confidence_pct:.1f}%\n"
            f"🕐 <b>Time:</b> {time_str}\n"
            f"📅 <b>Date:</b> {date_str}\n\n"
            f"<i>Audio monitoring system detected an important sound event.</i>"
        )
        
        return message
    
    def _get_sound_emoji(self, sound_type: str) -> str:
        """
        Get appropriate emoji for sound type.
        
        Args:
            sound_type: Type of sound
            
        Returns:
            Emoji string
        """
        sound_lower = sound_type.lower()
        
        if 'baby' in sound_lower or 'infant' in sound_lower or 'cry' in sound_lower:
            return "👶"
        elif 'glass' in sound_lower:
            return "🔨"
        elif 'fire' in sound_lower or 'alarm' in sound_lower:
            return "🔥"
        elif 'smoke' in sound_lower:
            return "💨"
        else:
            return "🔊"
    
    def send_status_message(self, message: str) -> bool:
        """
        Send a status message (bypasses cooldown).
        
        Args:
            message: Status message to send
            
        Returns:
            True if message was sent successfully
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success = loop.run_until_complete(self._send_message_async(message))
            loop.close()
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to send status message: {e}")
            return False
    
    def send_startup_message(self) -> bool:
        """Send a message when the monitoring system starts."""
        message = (
            "🟢 <b>Audio Monitoring Started</b>\n\n"
            "The audio monitoring system is now active and listening for:\n"
            "👶 Baby crying\n"
            "🔨 Glass breaking\n"
            "🔥 Fire alarms\n"
            "💨 Smoke detectors\n\n"
            "<i>You will receive notifications when these sounds are detected.</i>"
        )
        
        return self.send_status_message(message)
    
    def send_shutdown_message(self) -> bool:
        """Send a message when the monitoring system stops."""
        message = (
            "🔴 <b>Audio Monitoring Stopped</b>\n\n"
            "<i>The audio monitoring system has been stopped.</i>"
        )
        
        return self.send_status_message(message)
    
    def test_connection(self) -> bool:
        """
        Test the Telegram bot connection.
        
        Returns:
            True if connection is working, False otherwise
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Get bot info to test connection
            bot_info = loop.run_until_complete(self.bot.get_me())
            loop.close()
            
            self.logger.info(f"Telegram bot connection test successful: @{bot_info.username}")
            return True
            
        except Exception as e:
            self.logger.error(f"Telegram bot connection test failed: {e}")
            return False
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """
        Get notification statistics.
        
        Returns:
            Dictionary with notification stats
        """
        current_time = time.time()
        
        stats = {
            'total_sound_types': len(self.last_notification_times),
            'sound_types': list(self.last_notification_times.keys()),
            'last_notifications': {}
        }
        
        for sound_type, last_time in self.last_notification_times.items():
            time_since = current_time - last_time
            stats['last_notifications'][sound_type] = {
                'seconds_ago': time_since,
                'timestamp': datetime.fromtimestamp(last_time).isoformat()
            }
        
        return stats
