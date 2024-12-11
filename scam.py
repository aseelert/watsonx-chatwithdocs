import os
import pandas as pd
from sqlalchemy import create_engine, text
from langdetect import detect
from datetime import datetime
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Main directory for storing output files
output_directory = "/home/sharetribeisprettygood/python-scripts/scam"
backup_directory = os.path.join(output_directory, "backup")

# Ensure the output and backup directories exist
os.makedirs(output_directory, exist_ok=True)
os.makedirs(backup_directory, exist_ok=True)

# Database connection configuration
db_config = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'database': os.getenv('DB_NAME')
}

# Create SQLAlchemy engine
engine = create_engine(f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")

# Query to fetch messages containing '@gmail.com'
query_messages = "SELECT * FROM messages WHERE content LIKE '%@gmail.com%'"

# Query to fetch user details and join with emails table
query_users = """
SELECT p.id, p.username, p.given_name, p.family_name, p.created_at, p.email, p.google_oauth2_id, e.address AS email_address, p.last_sign_in_at
FROM people p
LEFT JOIN emails e ON p.id = e.person_id
"""

# Read data into a pandas DataFrame
df_messages = pd.read_sql(query_messages, engine)
df_users = pd.read_sql(query_users, engine)

# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        return "error"

# Apply language detection
df_messages['language'] = df_messages['content'].apply(detect_language)

# Filter for English content
english_messages = df_messages[df_messages['language'] == 'en'].copy()

# Convert created_at to datetime
english_messages.loc[:, 'created_at'] = pd.to_datetime(english_messages['created_at'])

# Identify potential scam messages
def identify_potential_scams(df):
    # Sort by sender_id and created_at
    df = df.sort_values(by=['sender_id', 'created_at'])

    # Group by sender_id and content to find duplicates within a short time frame
    scam_suspects = []
    for sender_id, group in df.groupby('sender_id'):
        group['time_diff'] = group['created_at'].diff().dt.total_seconds().fillna(0)
        group['is_scam'] = group.duplicated(subset=['content'], keep=False) & (group['time_diff'] <= 600)
        scam_suspects.append(group)

    scam_suspects_df = pd.concat(scam_suspects)
    return scam_suspects_df[scam_suspects_df['is_scam']]

scam_messages = identify_potential_scams(english_messages)

# Merge with user data to get user details
scam_messages = scam_messages.merge(df_users, left_on='sender_id', right_on='id', suffixes=('', '_user'))

# Write the potential scam messages to a CSV file
scam_messages.to_csv(os.path.join(output_directory, 'potential_scam_messages.csv'), index=False)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process scam messages.")
parser.add_argument('--ban-user', action='store_true', help="Execute the update_banned and update SQL scripts if scam messages are found.")
args = parser.parse_args()

# Only proceed if there are scam messages
if not scam_messages.empty:
    # Create the email content and SQL update commands
    email_content = ""
    sql_update_commands = []
    sql_select_commands = []
    sql_ban_commands = []
    sql_ban_select_commands = []
    new_content = (
        "Diese Nachricht wurde vom Administrator geändert, da sie als möglicherweise betrügerisch eingestuft wurde. "
        "This message has been changed by the administrator because it appears to be scam-related."
    )

    for _, row in scam_messages.iterrows():
        subject = f"Scam Alarm für Benutzer {row['sender_id']}, {row['username']}, {row['email']}, {row['content']}"
        email_body = (
            f"**Scam Alarm**\n\n"
            f"Es gibt einen möglichen Scam für den Benutzer {row['username']} (ID: {row['sender_id']}).\n\n"
            f"**Benutzerdetails**:\n\n"
            f"- Benutzer ID: {row['sender_id']}\n"
            f"- Benutzername: {row['username']}\n"
            f"- Email: {row['email']}\n\n"
            f"**Nachricht**:\n\n"
            f"```\n{row['content']}\n```\n"
        )
        email_content += f"Subject: {subject}\n\n{email_body}\n\n"

        # Backup current message content
        select_statement = text(f"SELECT content FROM messages WHERE id = {row['id']} AND sender_id = '{row['sender_id']}' AND created_at = '{row['created_at']}' AND conversation_id = '{row['conversation_id']}'")
        with engine.connect() as connection:
            current_content = connection.execute(select_statement).fetchone()

        if current_content:
            original_content = current_content[0]
            # Save the current message content in a backup file as an update statement
            sql_select_commands.append(
                f"UPDATE messages SET content = '{original_content}' WHERE id = {row['id']} AND sender_id = '{row['sender_id']}' AND created_at = '{row['created_at']}' AND conversation_id = '{row['conversation_id']}'"
            )
            # Update the message content with the new scam message notification
            sql_update_commands.append(
                f"UPDATE messages SET content = '{new_content}' WHERE id = {row['id']} AND sender_id = '{row['sender_id']}' AND created_at = '{row['created_at']}' AND conversation_id = '{row['conversation_id']}'"
            )

    # Extract unique sender_id values from scam_messages
    unique_sender_ids = scam_messages['sender_id'].unique()
    for sender_id in unique_sender_ids:
        # Backup current ban status
        select_statement = text(f"SELECT status FROM community_memberships WHERE person_id = '{sender_id}'")
        with engine.connect() as connection:
            current_status = connection.execute(select_statement).fetchone()

        if current_status:
            original_status = current_status[0]
            # Save the current status in a backup file as an update statement
            sql_ban_select_commands.append(
                f"UPDATE community_memberships SET status = '{original_status}' WHERE person_id = '{sender_id}'"
            )
            # Update the status to 'banned'
            sql_ban_commands.append(
                f"UPDATE community_memberships SET status = 'banned' WHERE person_id = '{sender_id}'"
            )

    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save backup files
    backup_subdir = os.path.join(backup_directory, timestamp)
    os.makedirs(backup_subdir, exist_ok=True)

    with open(os.path.join(backup_subdir, 'update_messages_backup.sql'), 'w') as f:
        f.write("\n".join(sql_select_commands))

    with open(os.path.join(backup_subdir, 'update_ban_backup.sql'), 'w') as f:
        f.write("\n".join(sql_ban_select_commands))

    # Save new update scripts for applying the changes
    update_messages_filename = os.path.join(output_directory, f'update_messages_{timestamp}.sql')
    update_ban_filename = os.path.join(output_directory, f'update_ban_{timestamp}.sql')

    with open(update_messages_filename, 'w') as f:
        f.write("\n".join(sql_update_commands))

    with open(update_ban_filename, 'w') as f:
        f.write("\n".join(sql_ban_commands))

    print(f"Backup and Banning SQL commands have been written to {backup_subdir}")

    # Save the email content to a file
    email_filename = os.path.join(output_directory, f'scam_alert_email_{timestamp}.txt')
    with open(email_filename, 'w') as f:
        f.write(email_content)

    # Execute the SQL scripts if --ban-user is used
    if args.ban_user:
        messages_executed = False
        ban_executed = False

        with engine.begin() as connection:  # Using begin() ensures a commit or rollback
            try:
                # Execute the update commands
                for statement in sql_update_commands:
                    connection.execute(text(statement))
                messages_executed = True

                # Execute the ban commands
                for statement in sql_ban_commands:
                    connection.execute(text(statement))
                ban_executed = True

            except Exception as e:
                print(f"An error occurred while executing SQL statements: {e}")

        if messages_executed:
            print(f"Messages update script '{update_messages_filename}' executed successfully.")
        else:
            print(f"Failed to execute messages update script '{update_messages_filename}'.")

        if ban_executed:
            print(f"Ban update script '{update_ban_filename}' executed successfully.")
        else:
            print(f"Failed to execute ban update script '{update_ban_filename}'.")

        # Verify if the user was banned successfully
        for sender_id in unique_sender_ids:
            with engine.connect() as connection:
                result = connection.execute(text(f"SELECT status FROM community_memberships WHERE person_id = '{sender_id}'")).fetchone()
                if result and result[0] == 'banned':
                    print(f"User with person_id '{sender_id}' was successfully banned.")
                else:
                    print(f"Failed to ban user with person_id '{sender_id}'. Current status: {result[0] if result else 'N/A'}")

    # Send the email using sendmail
    def send_email(recipient, subject, body):
        email_text = f"To: {recipient}\nSubject: {subject}\nMIME-Version: 1.0\nContent-Type: text/markdown\n\n{body}"
        with open("email.txt", "w") as f:
            f.write(email_text)
        os.system("sendmail -t < email.txt")

    with open(email_filename, 'r') as f:
        email_content = f.read()

    send_email("alexander.seelert@gmx.de", "Scam Alert Notification", email_content)
    print(f"Potential scam messages have been written to {output_directory} and files potential_scam_messages.csv, update_{timestamp}.sql, select_{timestamp}.sql created, and email sent to alexander.seelert@gmx.de")
else:
    print("No potential scam messages found.")
